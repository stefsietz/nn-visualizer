using UnityEngine;
using System.Collections.Generic;

public struct FeatureMapInfo
{
    /// <summary>
    /// 3d position of the featuremap
    /// </summary>
    public Vector3 position;

    /// <summary>
    /// 
    /// </summary>
    public Vector2Int shape;
    public Vector2Int outputShape;
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


    private Vector3 _outputPosition;
    private Vector2Int _shape;
    private Dictionary<Layer, Vector2Int> _convShapeForLayer = new Dictionary<Layer, Vector2Int>();
    private Dictionary<Layer, Vector2Int> _outputShapeForLayer;
    private Vector2 _theoreticalOutputShape;
    private int _index;

    public Dictionary<Layer, Vector2Int> stride = new Vector2Int(1, 1);

    /// <summary>
    /// Grid that provides the points for the pixels to be rendered
    /// </summary>
    private GridShape _pixelGrid;

    /// <summary>
    /// Grid that provides the poihnts for the outgoing connections
    /// </summary>
    private Dictionary<Layer, Shape> _filterGrid;

    /// <summary>
    /// Like Pixelgrid but can have a different stride / padding, it provides positions for the output kernel connections
    /// </summary>
    private Dictionary<Layer, GridShape> _outputGrid;
    private Dictionary<Layer, List<Shape>> _allCalcFilterGridsForLayer;

    public float spacing;

    public FeatureMap(I2DMapLayer layer, int index)
    {
        this._index = index;
        FeatureMapInfo info = layer.GetFeatureMapInfo(index);
        this.position = info.position;
        this._outputPosition = info.position;
        this._shape = info.shape;
        this._outputShape = info.outputShape;
        this.spacing = info.spacing;

        InitGrids();
    }

    public void AddOutputLayer(InputAcceptingLayer layer)
    {
        this._convShapeForLayer[layer] = layer.convShape;

        this.InitFilterGridsForLayer(layer);
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

                gr.spacing /= (_shape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);
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
            gr.spacing /= (_shape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

            GridShape gr2 = (GridShape)_allCalcFilterGridsForLayer[outputLayer][convLocation].Clone();
            gr2.spacing /= (_shape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

            filterGrids.Add(gr2); 
            return filterGrids;
        }
        else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach (GridShape gr in _allCalcFilterGridsForLayer[outputLayer])
            {

                gr.spacing /= (_shape.x - 1) / (float)(_convShapeForLayer[outputLayer].x - 1);

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

            return new GridShape(position, _shape, new Vector2(0, 0));
        }
        else if (allCalcs == 1.0f)
        {
            return _pixelGrid;
        }
        else
        {
            GridShape degenerate = new GridShape(position, _shape, new Vector2(0, 0));
            GridShape interpolated = degenerate.InterpolatedGrid(_pixelGrid, allCalcs);

            return interpolated;
        }
    }

    private void InitGrids()
    {
        _pixelGrid = new GridShape(position, _shape, Get2DSpacing());
        _outputGrid = new GridShape(_outputPosition, _shape, Get2DSpacing());
        _allCalcFilterGridsForLayer = new Dictionary<Layer, List<Shape>>();
    }

    private void InitFilterGridsForLayer(InputAcceptingLayer layer)
    {
        _allCalcFilterGridsForLayer[layer] = new List<Shape>();

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (_convShapeForLayer[layer].x > 1)
        {
            safeSpacing = (_shape.x - 1) / (float)(_convShapeForLayer[layer].x - 1) * Get2DSpacing();
        }

        _filterGrid = new GridShape(_outputPosition, _convShapeForLayer[layer], safeSpacing);

        Vector3[] allCalcPositions;

        if (_outputShape == _shape) //means stride == 1
        {
            allCalcPositions = _pixelGrid.GetVertices(true);
        }
        else
        {
            _outputGrid = new GridShape(_outputPosition, _outputShape, Get2DSpacing() * stride.x);
            allCalcPositions = _outputGrid.GetVertices(true);
        }

        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            _allCalcFilterGridsForLayer[layer].Add(new GridShape(allCalcPositions[i], _convShapeForLayer[layer], safeSpacing));
        }
    }

    private void UpdateGrids()
    {
        _pixelGrid.position = position;
        _pixelGrid.resolution = _shape;
        _pixelGrid.spacing = Get2DSpacing();

        _outputGrid.position = _outputPosition;
        _outputGrid.resolution = _outputShape;
        _outputGrid.spacing = Get2DSpacing() * stride;


        _filterGrid.position = _outputPosition;

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (_convShape.x > 1)
        {
            safeSpacing = (_shape.x - 1) / (float)(_convShape.x - 1) * Get2DSpacing();
        }

        ((GridShape)_filterGrid).spacing = safeSpacing;

        Vector3[] allCalcPositions;

        if (_outputShape == _shape) //means stride == 1
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
        if (this._shape != info.shape || this._convShape != info.convShape || this._outputShape != info.outputShape) reinit = true;

        this._shape = info.shape;
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