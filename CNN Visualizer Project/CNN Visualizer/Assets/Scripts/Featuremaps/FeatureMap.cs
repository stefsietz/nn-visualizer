using UnityEngine;
using System.Collections.Generic;

public struct FeatureMapOutputProperties
{
    public Vector2Int convShape;
    public Vector2Int outputFilterArrayShape;
    public Vector2 theoreticalOutputFilterArrayShape;
    public Vector2Int stride;
    public Vector2Int dilution;

    public Vector3 positionOffset;

    /// <summary>
    /// Grid that provides the points for the outgoing connections
    /// </summary>
    public GridShape filterGrid;

    /// <summary>
    /// Like Pixelgrid but can have a different stride / padding, it provides positions for the output kernel connections
    /// </summary>
    public GridShape filterInstanceGrid;

    /// <summary>
    /// List of filter grids, at positions of filterInstanceGrid.
    /// </summary>
    public List<GridShape> allCalcFilterGrids;
}

public struct FeatureMapInputProperties
{
    /// <summary>
    /// 3d position of the featuremap
    /// </summary>
    public Vector3 position;

    /// <summary>
    /// input resolution of the featuremap
    /// </summary>
    public Vector2Int inputShape;
    public float spacing;
}

public interface I2DMapLayer
{
    FeatureMapInputProperties GetFeatureMapInputProperties(int featureMapIndex);
}
public class FeatureMap
{
    /// <summary>
    /// 3d position of the featuremap
    /// </summary>
    public Vector3 position;


    private Vector2Int _inputShape;

    private int _index;

    /// <summary>
    /// Grid that provides the points for the pixels to be rendered
    /// </summary>
    private GridShape _pixelGrid;

    /// <summary>
    /// Per Outputlayer Information
    /// </summary>
    private Dictionary<InputAcceptingLayer, FeatureMapOutputProperties> _outputPropertiesDict = new Dictionary<InputAcceptingLayer, FeatureMapOutputProperties>();

    public float spacing;

    public FeatureMap(I2DMapLayer layer, int index)
    {
        this._index = index;
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(index);
        this.position = info.position;
        this._inputShape = info.inputShape;
        this.spacing = info.spacing;

        InitGrids();
    }

    public void AddOutputLayer(InputAcceptingLayer outputLayer)
    {
        if (!this._outputPropertiesDict.ContainsKey(outputLayer))
            this._outputPropertiesDict[outputLayer] = new FeatureMapOutputProperties();

        FeatureMapOutputProperties props = this._outputPropertiesDict[outputLayer];
        props.convShape = outputLayer.convShape;
        props.outputFilterArrayShape = GetFeatureMapShapeFromInput(this._inputShape, outputLayer.convShape, outputLayer.stride, outputLayer.GetPadding());
        props.theoreticalOutputFilterArrayShape = GetTheoreticalFloatFeatureMapShapeFromInput(this._inputShape, outputLayer.convShape, outputLayer.stride, outputLayer.GetPadding());

        props.stride = outputLayer.stride;
        props.dilution = outputLayer.dilution;

        props.positionOffset = GetOutputGridOffset(props.theoreticalOutputFilterArrayShape, props.outputFilterArrayShape);

        this._outputPropertiesDict[outputLayer] = props;

        this.InitOrUpdateOutputGridsForLayer(outputLayer, true);
    }

    public void RemoveOutputLayer(InputAcceptingLayer outputLayer)
    {
        _outputPropertiesDict.Remove(outputLayer);
    }

    public GridShape GetPixelGrid()
    {
        return _pixelGrid;
    }

    /// <summary>
    /// Returns Grids in the shape of the conv filter, usually serving as out connections of the according layer.
    /// </summary>
    /// <param name="outputLayer"> Connected Layer that needs the input conv filter points for drawing the connections.
    /// <param name="outputShape">Calculated integer 2d featuremap shape of this layer, taking into account stride and padding.</param>
    /// <param name="theoreticalOutputShape">Calculated float 2d featuremap shape of this layer, taking into account stride and padding. Can contain fractional part because of stride division.</param>
    /// <param name="stride"></param>
    /// <param name="allCalcs">Interpolation parameter for all calc view</param>
    /// <returns></returns>
    public List<Shape> GetFilterGrids(InputAcceptingLayer outputLayer, float allCalcs)
    {
        FeatureMapOutputProperties props = _outputPropertiesDict[outputLayer];

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            filterGrids.Add(props.filterGrid);
            return filterGrids;
        } else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach(GridShape gr in props.allCalcFilterGrids)
            {

                gr.spacing /= (_inputShape.x - 1) / (float)(props.convShape.x - 1);
                GridShape interpolated = gr.InterpolatedGrid(((GridShape)props.filterGrid), 1.0f - allCalcs);
                filterGrids.Add(interpolated);
            }
            return filterGrids;
        }
    }

    public List<Shape> GetFilterGrids(InputAcceptingLayer outputLayer, float allCalcs, int convLocation)
        //TODO: maybe rename as "GetFilterGridsForOutputStartpoints"?
    {
        FeatureMapOutputProperties props = _outputPropertiesDict[outputLayer];

        if(convLocation == -1)
        {
            return GetFilterGrids(outputLayer, allCalcs);
        }

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            GridShape gr = (GridShape)props.allCalcFilterGrids[convLocation].Clone();
            gr.spacing /= (_inputShape.x - 1) / (float)(props.convShape.x - 1);

            GridShape gr2 = (GridShape)props.allCalcFilterGrids[convLocation].Clone();
            gr2.spacing /= (_inputShape.x - 1) / (float)(props.convShape.x - 1);

            filterGrids.Add(gr2); 
            return filterGrids;
        }
        else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach (GridShape gr in props.allCalcFilterGrids)
            {

                gr.spacing /= (_inputShape.x - 1) / (float)(props.convShape.x - 1);

                GridShape gr2 = (GridShape)props.allCalcFilterGrids[convLocation].Clone();

                GridShape interpolated = gr.InterpolatedGrid(gr2, 1.0f - allCalcs);
                filterGrids.Add(interpolated);
            }
            return filterGrids;
        }
    }

    private Vector3 GetOutputGridOffset(Vector2 theoreticalOutputShape, Vector2Int outputShape)
    {
        Vector2 safeOffset = new Vector2(0, 0);
        if (theoreticalOutputShape != outputShape)
        {
            if (outputShape.x > 1 && outputShape.y > 1)
                safeOffset = (Get2DSpacing()) * 0.5f;
        }
        return new Vector3(safeOffset.x, safeOffset.y, 0);
    }

    /// <summary>
    /// provides line endpoints for input layer
    /// </summary>
    /// <param name="allCalcs"></param>
    /// <returns></returns>
    public Shape GetInputGrid(float allCalcs) // TODO: name is not very clear, maybe name "GetGridForInputEndpoints" or smth similar?
    {
        if (allCalcs == 0)
        {

            return new GridShape(position, _inputShape, new Vector2(0, 0));
        }
        else if (allCalcs == 1.0f)
        {
            return _pixelGrid;
        }
        else
        {
            GridShape degenerate = new GridShape(position, _inputShape, new Vector2(0, 0));
            GridShape interpolated = degenerate.InterpolatedGrid(_pixelGrid, allCalcs);

            return interpolated;
        }
    }

    private void InitGrids()
    {
        _pixelGrid = new GridShape(position, _inputShape, Get2DSpacing());
    }

    private void InitOrUpdateOutputGridsForLayer(InputAcceptingLayer outputLayer, bool initialize)
    {
        FeatureMapOutputProperties props = this._outputPropertiesDict[outputLayer];
        props.filterInstanceGrid = InitOrUpdateGrid(initialize ? null : props.filterInstanceGrid, position + props.positionOffset,
            props.outputFilterArrayShape, Get2DSpacing() * props.stride.x); //TODO: 2 Dimensional stride / spacing

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (props.convShape.x > 1)
        {
            safeSpacing = (_inputShape.x - 1) / (float)(props.convShape.x - 1) * Get2DSpacing(); //TODO: 2 Dimensional?
        }

        Vector3[] allCalcPositions = props.filterInstanceGrid.GetVertices(true);

        props.filterGrid = InitOrUpdateGrid(initialize ? null : props.filterGrid, position + props.positionOffset, props.convShape, safeSpacing);

        if(initialize)
            props.allCalcFilterGrids = new List<GridShape>();

        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            GridShape currentGrid = InitOrUpdateGrid(initialize ? null : props.allCalcFilterGrids[i], allCalcPositions[i], props.convShape, safeSpacing);
            if (initialize)
                props.allCalcFilterGrids.Add(currentGrid);
            else
                props.allCalcFilterGrids[i] = props.allCalcFilterGrids[i];
        }

        this._outputPropertiesDict[outputLayer] = props;
    }

    /// <summary>
    /// Initializes or updates a grid with provided parameters and then returns it. If grid is null, a new instance is created.
    /// </summary>
    /// <param name="grid"></param>
    /// <param name="position"></param>
    /// <param name="shape"></param>
    /// <param name="spacing"></param>
    /// <returns></returns>
    private GridShape InitOrUpdateGrid(GridShape grid, Vector3 position, Vector2Int shape, Vector2 spacing)
    {
        GridShape outgrid = grid;
        if (outgrid == null)
        {
            outgrid = new GridShape(position, shape, spacing);
        }
        else
        {
            outgrid.position = position;
            outgrid.resolution = shape;
            outgrid.spacing = spacing;
        }

        return outgrid;
    }


    /// <summary>
    /// Only for updates that don't change Geomtetry, after shape changes call ReInitValues instead
    /// </summary>
    /// <param name="layer"></param>
    public void UpdateValuesForInputParams(I2DMapLayer layer)
    {
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(_index);
        this.position = info.position;
        this.spacing = info.spacing;
    }

    public void ReInitForChangedInputParams(I2DMapLayer layer)
    {
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(_index);
        this.position = info.position;
        this.spacing = info.spacing;
        this._inputShape = info.inputShape;

        InitGrids();
    }

    public void UpdateForChangedOutputParams(InputAcceptingLayer outputLayer)
    {
        InitOrUpdateOutputGridsForLayer(outputLayer, false);
    }

    public void ReInitForChangedOutputParams(InputAcceptingLayer outputLayer)
    {
        InitOrUpdateOutputGridsForLayer(outputLayer, true);
    }

    public static Vector2Int GetFeatureMapShapeFromInput(Vector2Int inputShape, Vector2Int convShape, Vector2Int inputStride, Vector2Int padding)
    {
        Vector2 featureMapDims = (inputShape - convShape + new Vector2(2f, 2f) * padding) / (Vector2) inputStride + new Vector2(1f, 1f);
        Vector2Int intFeatureMapDims = Vector2Int.FloorToInt(featureMapDims);

        return intFeatureMapDims;
    }

    public static Vector2 GetTheoreticalFloatFeatureMapShapeFromInput(Vector2Int inputShape, Vector2Int convShape, Vector2Int inputStride, Vector2Int padding)
    {
        Vector2 featureMapDims = (inputShape - convShape + new Vector2(2f, 2f) * padding) / inputStride + new Vector2(1f, 1f);

        return featureMapDims;
    }

    private Vector2 Get2DSpacing()
    {
        return new Vector2(spacing, spacing);
    }
}