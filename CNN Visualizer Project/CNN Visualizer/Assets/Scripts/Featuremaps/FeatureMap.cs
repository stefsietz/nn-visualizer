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

public struct FeatureMapInfo
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
    FeatureMapInfo GetFeatureMapInfo(int featureMapIndex);
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
        FeatureMapInfo info = layer.GetFeatureMapInfo(index);
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

        this.InitFilterGridsForLayer(outputLayer);
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
    public List<Shape> GetFilterGrids(InputAcceptingLayer outputLayer, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs)
    {
        ReinitGridsIfNecessary(outputLayer, outputShape, theoreticalOutputShape, stride);

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            filterGrids.Add(_filterGrid);
            return filterGrids;
        } else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach(GridShape gr in _allCalcFilterGridsForLayer[outputLayer])
            {

                gr.spacing /= (_inputShape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);
                GridShape interpolated = gr.InterpolatedGrid(((GridShape)_filterGrid), 1.0f - allCalcs);
                filterGrids.Add(interpolated);
            }
            return filterGrids;
        }
    }

    public List<Shape> GetFilterGrids(InputAcceptingLayer outputLayer, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs, int convLocation)
        //TODO: maybe rename as "GetFilterGridsForOutputStartpoints"?
    {
        if(convLocation == -1)
        {
            return GetFilterGrids(outputLayer, outputShape, theoreticalOutputShape, stride, allCalcs);
        }

        ReinitGridsIfNecessary(outputLayer, outputShape, theoreticalOutputShape, stride);

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            GridShape gr = (GridShape)_allCalcFilterGridsForLayer[outputLayer][convLocation].Clone();
            gr.spacing /= (_inputShape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

            GridShape gr2 = (GridShape)_allCalcFilterGridsForLayer[outputLayer][convLocation].Clone();
            gr2.spacing /= (_inputShape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

            filterGrids.Add(gr2); 
            return filterGrids;
        }
        else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach (GridShape gr in _allCalcFilterGridsForLayer[outputLayer])
            {

                gr.spacing /= (_inputShape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

                GridShape gr2 = (GridShape)_allCalcFilterGridsForLayer[outputLayer][convLocation].Clone();

                GridShape interpolated = gr.InterpolatedGrid(gr2, 1.0f - allCalcs);
                filterGrids.Add(interpolated);
            }
            return filterGrids;
        }
    }

    private void ReinitGridsIfNecessary(InputAcceptingLayer outputLayer, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride)
    {
        //check if requested outputshape is same as existing, reinit allcalgrids if not
        if (outputShape != this._outputShape
            || stride != this.stride
            || this._theoreticalOutputShape != theoreticalOutputShape
            && outputShape != new Vector2Int(0, 0))
        {
            this._outputShape = outputShape;
            this._theoreticalOutputShape = theoreticalOutputShape;
            this._outputPosition = position + GetOutputGridOffset(theoreticalOutputShape, outputShape);
            this.stride = stride;
            InitGrids();
            InitFilterGridsForLayer(outputLayer);
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
        _outputGrid = new GridShape(_outputPosition, _inputShape, Get2DSpacing());
        _allCalcFilterGridsForLayer = new Dictionary<Layer, List<Shape>>();
    }

    private void InitFilterGridsForLayer(InputAcceptingLayer outputLayer)
    {
        FeatureMapOutputProperties props = this._outputPropertiesDict[outputLayer];
        props.filterInstanceGrid = new GridShape(position + props.positionOffset, props.outputFilterArrayShape, Get2DSpacing() * props.stride.x); //TODO: 2 Dimensional stride / spacing

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (props.convShape.x > 1)
        {
            safeSpacing = (_inputShape.x - 1) / (float)(props.convShape.x - 1) * Get2DSpacing(); //TODO: 2 Dimensional?
        }

        Vector3[] allCalcPositions = props.filterInstanceGrid.GetVertices(true);

        props.allCalcFilterGrids = new List<GridShape>();
        props.filterGrid = new GridShape(position + props.positionOffset, props.convShape, safeSpacing);


        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            props.allCalcFilterGrids.Add(new GridShape(allCalcPositions[i], props.convShape, safeSpacing));
        }

        this._outputPropertiesDict[outputLayer] = props;
    }

    private void UpdateGrids()
    {
        _pixelGrid.position = position;
        _pixelGrid.resolution = _inputShape;
        _pixelGrid.spacing = Get2DSpacing();

        _outputGrid.position = _outputPosition;
        _outputGrid.resolution = _outputShape;
        _outputGrid.spacing = Get2DSpacing() * stride;


        _filterGrid.position = _outputPosition;

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (_convShape.x > 1)
        {
            safeSpacing = (_inputShape.x - 1) / (float)(_convShape.x - 1) * Get2DSpacing();
        }

        ((GridShape)_filterGrid).spacing = safeSpacing;

        Vector3[] allCalcPositions;

        if (_outputShape == _inputShape) //means stride == 1
        {
            allCalcPositions = _pixelGrid.GetVertices(true);
        }
        else
        {
            _outputGrid.position = _outputPosition + new Vector3(0, 1.0f, 0);
            _outputGrid.resolution = _outputShape;
            _outputGrid.spacing = safeSpacing;

            allCalcPositions = _outputGrid.GetVertices(true);
        }

        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            ((GridShape)_allCalcFilterGrids[i]).position = allCalcPositions[i];
            ((GridShape)_allCalcFilterGrids[i]).resolution = _convShape;
            ((GridShape)_allCalcFilterGrids[i]).spacing = safeSpacing;
        }
    }

    public void UpdateValues(I2DMapLayer layer)
    {
        FeatureMapInfo info = layer.GetFeatureMapInfo(_index);
        this.position = info.position;
        this._outputPosition = info.position;
        this.spacing = info.spacing;




        bool reinit = false;
        if (this._inputShape != info.inputShape || this._convShape != info.convShape || this._outputShape != info.outputShape) reinit = true;

        this._inputShape = info.inputShape;
        this._outputShape = info.outputShape;
        this._convShape = info.convShape;

        if (reinit) InitGrids();
        else UpdateGrids();
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