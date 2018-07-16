using UnityEngine;
using System.Collections.Generic;

public struct FeatureMapInfo
{
    public Vector3 position;
    public Vector2Int shape;
    public Vector2Int filterShape;
    public Vector2Int outputShape;
    public float spacing;
}

public interface I2DMapLayer
{
    FeatureMapInfo GetFeatureMapInfo(int featureMapIndex);
}
public class FeatureMap
{
    public Vector3 position;
    private Vector3 outputPosition;
    public Vector2Int shape;
    public Vector2Int filterShape;
    public Vector2Int outputShape;
    private Vector2 theoreticalOutputShape;
    private int index;

    public Vector2Int stride = new Vector2Int(1, 1);

    private GridShape _pixelGrid;
    private Shape _filterGrid;
    private GridShape _outputGrid;
    private List<Shape> _allCalcFilterGrids;

    public float spacing;

    public FeatureMap(I2DMapLayer layer, int index)
    {
        this.index = index;
        FeatureMapInfo info = layer.GetFeatureMapInfo(index);
        this.position = info.position;
        this.outputPosition = info.position;
        this.shape = info.shape;
        this.outputShape = info.outputShape;
        this.filterShape = info.filterShape;
        this.spacing = info.spacing;

        InitGrids();
    }

    public GridShape GetPixelGrid()
    {
        return _pixelGrid;
    }

    public List<Shape> GetFilterGrids(Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs)
    {
        //check if requested outputshape is same as existing, reinit allcalgrids if not
        if(outputShape != this.outputShape
            ||  stride != this.stride
            || this.theoreticalOutputShape != theoreticalOutputShape
            && outputShape != new Vector2Int(0, 0))
        {
            this.outputShape = outputShape;
            this.theoreticalOutputShape = theoreticalOutputShape;
            this.outputPosition = position + GetOutputGridOffset(theoreticalOutputShape, outputShape);
            this.stride = stride;
            InitGrids();
        }

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            filterGrids.Add(_filterGrid);
            return filterGrids;
        } else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach(GridShape gr in _allCalcFilterGrids)
            {

                gr.spacing /= (shape.x - 1) / (float)(filterShape.x - 1);
                GridShape interpolated = gr.InterpolatedGrid(((GridShape)_filterGrid), 1.0f - allCalcs);
                filterGrids.Add(interpolated);
            }
            return filterGrids;
        }
    }

    public List<Shape> GetFilterGrids(Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs, int convLocation)
    {
        if(convLocation == -1)
        {
            return GetFilterGrids(outputShape, theoreticalOutputShape, stride, allCalcs);
        }

        //check if requested outputshape is same as existing, reinit allcalgrids if not
        if (outputShape != this.outputShape
            || stride != this.stride
            || this.theoreticalOutputShape != theoreticalOutputShape
            && outputShape != new Vector2Int(0, 0))
        {
            this.outputShape = outputShape;
            this.theoreticalOutputShape = theoreticalOutputShape;
            this.outputPosition = position + GetOutputGridOffset(theoreticalOutputShape, outputShape);
            this.stride = stride;
            InitGrids();
        }

        if (allCalcs == 0)
        {
            List<Shape> filterGrids = new List<Shape>();
            GridShape gr = (GridShape)_allCalcFilterGrids[convLocation].Clone();
            gr.spacing /= (shape.x - 1) / (float)(filterShape.x - 1);

            GridShape gr2 = (GridShape)_allCalcFilterGrids[convLocation].Clone();
            gr2.spacing /= (shape.x - 1) / (float)(filterShape.x - 1);

            filterGrids.Add(gr2); 
            return filterGrids;
        }
        else
        {
            List<Shape> filterGrids = new List<Shape>();
            foreach (GridShape gr in _allCalcFilterGrids)
            {

                gr.spacing /= (shape.x - 1) / (float)(filterShape.x - 1);

                GridShape gr2 = (GridShape)_allCalcFilterGrids[convLocation].Clone();

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

    public Shape GetInputGrid(float allCalcs)
    {
        if (allCalcs == 0)
        {

            return new GridShape(position, shape, new Vector2(0, 0));
        }
        else if (allCalcs == 1.0f)
        {
            return _pixelGrid;
        }
        else
        {
            GridShape degenerate = new GridShape(position, shape, new Vector2(0, 0));
            GridShape interpolated = degenerate.InterpolatedGrid(_pixelGrid, allCalcs);

            return interpolated;
        }
    }

    private void InitGrids()
    {
        _pixelGrid = new GridShape(position, shape, Get2DSpacing());
        _outputGrid = new GridShape(outputPosition, shape, Get2DSpacing());
        _allCalcFilterGrids = new List<Shape>();

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if(filterShape.x > 1)
        {
            safeSpacing = (shape.x - 1) / (float)(filterShape.x - 1) * Get2DSpacing();
        }

        _filterGrid = new GridShape(outputPosition, filterShape, safeSpacing); 

        Vector3[] allCalcPositions;
      
        if(outputShape == shape) //means stride == 1
        {
            allCalcPositions = _pixelGrid.GetVertices(true);
        } else
        {
            _outputGrid = new GridShape(outputPosition, outputShape, Get2DSpacing() * stride.x);
            allCalcPositions = _outputGrid.GetVertices(true);
        }

        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            _allCalcFilterGrids.Add(new GridShape(allCalcPositions[i], filterShape, safeSpacing));
        }

    }

    private void UpdateGrids()
    {
        _pixelGrid.position = position;
        _pixelGrid.resolution = shape;
        _pixelGrid.spacing = Get2DSpacing();

        _outputGrid.position = outputPosition;
        _outputGrid.resolution = outputShape;
        _outputGrid.spacing = Get2DSpacing() * stride;


        _filterGrid.position = outputPosition;

        Vector2 safeSpacing = new Vector2(0.0f, 0.0f);
        if (filterShape.x > 1)
        {
            safeSpacing = (shape.x - 1) / (float)(filterShape.x - 1) * Get2DSpacing();
        }

        ((GridShape)_filterGrid).spacing = safeSpacing;

        Vector3[] allCalcPositions;

        if (outputShape == shape) //means stride == 1
        {
            allCalcPositions = _pixelGrid.GetVertices(true);
        }
        else
        {
            _outputGrid.position = outputPosition;
            _outputGrid.resolution = outputShape;
            _outputGrid.spacing = safeSpacing;

            allCalcPositions = _outputGrid.GetVertices(true);
        }

        for (int i = 0; i < allCalcPositions.Length; i++)
        {
            ((GridShape)_allCalcFilterGrids[i]).position = allCalcPositions[i];
            ((GridShape)_allCalcFilterGrids[i]).resolution = filterShape;
            ((GridShape)_allCalcFilterGrids[i]).spacing = safeSpacing;
        }
    }

    public void UpdateValues(I2DMapLayer layer)
    {
        FeatureMapInfo info = layer.GetFeatureMapInfo(index);
        this.position = info.position;
        this.outputPosition = info.position;
        this.spacing = info.spacing;




        bool reinit = false;
        if (this.shape != info.shape || this.filterShape != info.filterShape || this.outputShape != info.outputShape) reinit = true;

        this.shape = info.shape;
        this.outputShape = info.outputShape;
        this.filterShape = info.filterShape;

        if (reinit) InitGrids();
        else UpdateGrids();
    }

    public static Vector2Int GetFeatureMapShapeFromInput(Vector2Int inputShape, Vector2Int filterShape, Vector2Int inputStride, Vector2Int padding)
    {
        Vector2 featureMapDims = (inputShape - filterShape + new Vector2(2f, 2f) * padding) / (Vector2) inputStride + new Vector2(1f, 1f);
        Vector2Int intFeatureMapDims = Vector2Int.FloorToInt(featureMapDims);

        return intFeatureMapDims;
    }

    public static Vector2 GetTheoreticalFloatFeatureMapShapeFromInput(Vector2Int inputShape, Vector2Int filterShape, Vector2Int inputStride, Vector2Int padding)
    {
        Vector2 featureMapDims = (inputShape - filterShape + new Vector2(2f, 2f) * padding) / inputStride + new Vector2(1f, 1f);

        return featureMapDims;
    }

    private Vector2 Get2DSpacing()
    {
        return new Vector2(spacing, spacing);
    }
}