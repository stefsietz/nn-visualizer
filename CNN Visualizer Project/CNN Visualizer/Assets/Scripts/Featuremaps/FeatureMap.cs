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
    private Vector3 _position;
    private float _spacing;
    private Vector2Int _inputShape;

    /// <summary>
    /// index of this featuremap in the owning layer
    /// </summary>
    private int _index;

    /// <summary>
    /// Grid that provides the points for the pixels to be rendered
    /// </summary>
    private GridShape _pixelGrid;

    /// <summary>
    /// Per Outputlayer Information
    /// </summary>
    private Dictionary<InputAcceptingLayer, FeatureMapOutputProperties> _outputPropertiesDict = new Dictionary<InputAcceptingLayer, FeatureMapOutputProperties>();

    public FeatureMap(I2DMapLayer layer, int index)
    {
        this._index = index;
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(index);
        this._position = info.position;
        this._inputShape = info.inputShape;
        this._spacing = info.spacing;

        InitPixelGrid();
    }

    private void InitPixelGrid()
    {
        _pixelGrid = new GridShape(_position, _inputShape, Get2DSpacing());
    }

    public void AddOutputLayer(InputAcceptingLayer outputLayer)
    {
        if (!this._outputPropertiesDict.ContainsKey(outputLayer))
            this._outputPropertiesDict[outputLayer] = new FeatureMapOutputProperties();

        this._outputPropertiesDict[outputLayer] = FillOutputPropsFromLayer(outputLayer);

        this.InitOrUpdateOutputGridsForLayer(outputLayer, true);
    }

    public void UpdateForOutputLayer(InputAcceptingLayer outputLayer, bool topoChanged)
    {
        if (!this._outputPropertiesDict.ContainsKey(outputLayer))
        {
            Debug.Log("Keys: \n");
            foreach (InputAcceptingLayer key in this._outputPropertiesDict.Keys)
            {
                Debug.Log(key.name);
            }
            throw new System.Exception("OutputLayer has not yet been added to FeatureMap!");
        }

        this._outputPropertiesDict[outputLayer] = FillOutputPropsFromLayer(outputLayer);

        this.InitOrUpdateOutputGridsForLayer(outputLayer, topoChanged);
    }

    private FeatureMapOutputProperties FillOutputPropsFromLayer(InputAcceptingLayer outputLayer)
    {
        FeatureMapOutputProperties props = this._outputPropertiesDict[outputLayer];
        props.convShape = outputLayer.convShape;
        if (outputLayer.Is2dLayer())
        {
            props.outputFilterArrayShape = GetFeatureMapShapeFromInput(this._inputShape, outputLayer.convShape, outputLayer.stride, outputLayer.GetPadding());
            props.theoreticalOutputFilterArrayShape = GetTheoreticalFloatFeatureMapShapeFromInput(this._inputShape, outputLayer.convShape, outputLayer.stride, outputLayer.GetPadding());
            props.positionOffset = GetOutputGridOffset(props.theoreticalOutputFilterArrayShape, props.outputFilterArrayShape);
        }
        else
        {
            props.outputFilterArrayShape = this._inputShape;
            props.theoreticalOutputFilterArrayShape = this._inputShape;
            props.positionOffset = new Vector3(0, 0, 0);
        }

        props.stride = outputLayer.stride;
        props.dilution = outputLayer.dilution;
        return props;
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
    public List<GridShape> GetFilterGrids(InputAcceptingLayer outputLayer, float allCalcs)
    {
        FeatureMapOutputProperties props = _outputPropertiesDict[outputLayer];

        if (allCalcs == 0)
        {
            List<GridShape> filterGrids = new List<GridShape>();
            filterGrids.Add(props.filterGrid);
            return filterGrids;
        } else
        {
            return GetAllCalcFilterGrids(allCalcs, props);
        }
    }

    private List<GridShape> GetAllCalcFilterGrids(float allCalcs, FeatureMapOutputProperties props)
    {
        List<GridShape> filterGrids = new List<GridShape>();
        foreach (GridShape acFilterGrid in props.allCalcFilterGrids)
        {

            SetFilterGridSpacingToPreservePixelWidth(acFilterGrid, _inputShape, props.convShape);
            GridShape interpolated = acFilterGrid.InterpolatedGrid(((GridShape)props.filterGrid), 1.0f - allCalcs);
            filterGrids.Add(interpolated);
        }
        return filterGrids;
    }

    private void SetFilterGridSpacingToPreservePixelWidth(GridShape grid, Vector2Int inputShape, Vector2Int convShape)
    {
        grid.spacing /= (_inputShape.x - 1) / (float)(convShape.x - 1);
    }

    public List<GridShape> GetFilterGrids(InputAcceptingLayer outputLayer, float allCalcs, int convLocation)
        //TODO: maybe rename as "GetFilterGridsForOutputStartpoints"?
    {
        FeatureMapOutputProperties props = _outputPropertiesDict[outputLayer];

        if(convLocation == -1)
        {
            return GetFilterGrids(outputLayer, allCalcs);
        }

        if (allCalcs == 0)
        {
            List<GridShape> filterGrids = new List<GridShape>();
            GridShape gr = (GridShape)props.allCalcFilterGrids[convLocation].Clone();
            SetFilterGridSpacingToPreservePixelWidth(gr, _inputShape, props.convShape);

            GridShape gr2 = (GridShape)props.allCalcFilterGrids[convLocation].Clone();
            SetFilterGridSpacingToPreservePixelWidth(gr2, _inputShape, props.convShape);

            filterGrids.Add(gr2); 
            return filterGrids;
        }
        else
        {
            List<GridShape> filterGrids = new List<GridShape>();
            foreach (GridShape gr in props.allCalcFilterGrids)
            {

                SetFilterGridSpacingToPreservePixelWidth(gr, _inputShape, props.convShape);

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
    public Shape GetGridForInputEndpoints(float allCalcs) // TODO: name is not very clear, maybe name "GetGridForInputEndpoints" or smth similar?
    {
        if (allCalcs == 0)
        {

            return new GridShape(_position, _inputShape, new Vector2(0, 0));
        }
        else if (allCalcs == 1.0f)
        {
            return _pixelGrid;
        }
        else
        {
            GridShape degenerate = new GridShape(_position, _inputShape, new Vector2(0, 0));
            GridShape interpolated = degenerate.InterpolatedGrid(_pixelGrid, allCalcs);

            return interpolated;
        }
    }

    private void InitOrUpdateOutputGridsForLayer(InputAcceptingLayer outputLayer, bool initialize)
    {
        FeatureMapOutputProperties props = this._outputPropertiesDict[outputLayer];
        props.filterInstanceGrid = InitOrUpdateGrid(initialize ? null : props.filterInstanceGrid, _position + props.positionOffset,
            props.outputFilterArrayShape, Get2DSpacing() * props.stride.x); //TODO: 2 Dimensional stride / spacing

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (props.convShape.x > 1)
        {
            safeSpacing = (_inputShape.x - 1) / (float)(props.convShape.x - 1) * Get2DSpacing(); //TODO: 2 Dimensional?
        } else
        {
            safeSpacing = Get2DSpacing();
        }

        Vector3[] allCalcPositions = props.filterInstanceGrid.GetVertices(true);

        props.filterGrid = InitOrUpdateGrid(initialize ? null : props.filterGrid, _position + props.positionOffset, props.convShape, safeSpacing);

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
    public void UpdateValuesForInputParams(I2DMapLayer layer, bool topoChagned)
    {
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(_index);
        this._position = info.position;
        this._spacing = info.spacing;

        InitOrUpdateGrid(_pixelGrid, _position, _inputShape, Get2DSpacing());

        if (topoChagned)
        {
            this._inputShape = info.inputShape;
            _pixelGrid.InitVerts();
        }
    }

    public void ReInitForChangedInputParams(I2DMapLayer layer)
    {
        FeatureMapInputProperties info = layer.GetFeatureMapInputProperties(_index);
        this._position = info.position;
        this._spacing = info.spacing;
        this._inputShape = info.inputShape;

        InitPixelGrid();
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
        Vector2 featureMapDims = GetTheoreticalFloatFeatureMapShapeFromInput(inputShape, convShape, inputStride, padding);
        Vector2Int intFeatureMapDims = Vector2Int.FloorToInt(featureMapDims);

        return intFeatureMapDims;
    }

    public static Vector2 GetTheoreticalFloatFeatureMapShapeFromInput(Vector2Int inputShape, Vector2Int convShape, Vector2Int inputStride, Vector2Int padding)
    {
        Vector2 featureMapDims = (inputShape - convShape + new Vector2(2f, 2f) * padding) / inputStride + new Vector2(1f, 1f);

        return featureMapDims;
    }

    /// <summary>
    /// Returns a Vector2 with the spacing member set to both coordinates
    /// </summary>
    /// <returns></returns>
    private Vector2 Get2DSpacing()
    {
        return new Vector2(_spacing, _spacing);
    }

    public Vector3 GetPosition()
    {
        return _position;
    }
}