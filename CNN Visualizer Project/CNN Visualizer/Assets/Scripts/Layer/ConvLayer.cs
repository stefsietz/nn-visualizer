using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// Represents a Convolutional Layer of a CNN.
/// </summary>
public class ConvLayer : InputAcceptingLayer
{
    protected int _oldFullDepth = 64;
    public int fullDepth;

    /// <summary>
    /// 
    /// </summary>
    [Range(0.0f, 1.0f)]
    public float allFiltersSpacing = 0.5f;

    /// <summary>
    /// can iterate through different kernel positions on the 2d featuremap. -1 to display the kernel spread out over the whole featuremap
    /// </summary>
    [Range(-1, 128)]
    public int convLocation = 0;

    public float fullResHeight = 5.0f;

    public float nodeSize;

    protected Dictionary<int, Array> _weightTensorPerEpoch = new Dictionary<int, Array>();
    protected int[] _weightTensorShape = new int[4];

    protected Dictionary<int, Array> _activationTensorPerEpoch = new Dictionary<int, Array>();
    protected int[] _activationTensorShape = new int[4];

    public ConvLayer()
    {
        convShape = new Vector2Int(3, 3);
        _oldPadding = padding;
    }

    /// <summary>
    /// Checks if parameters have been updated and reinitializes necessary data.
    /// </summary>
    protected override void UpdateForChangedParams(bool topoChanged)
    {
        if (_featureMaps == null)
        {
            InitFeatureMapsForInputParams();
        }

        if (_inputLayer)
        {
            SetupFeaturemapResolution();
        }

        base.UpdateForChangedParams(topoChanged);
    }

    protected virtual void SetupFeaturemapResolution()
    {
        _featureMapResolution = FeatureMap.GetFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
        _featureMapTheoreticalResolution = FeatureMap.GetTheoreticalFloatFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
    }

    /// <summary>
    /// Returns a List of Shapes representing the pixels of the feature maps.
    /// </summary>
    /// <returns></returns>
    protected override List<GridShape> GetPointShapes()
    {
        UpdateFeatureMapsForInputParams(true);

        List<GridShape> pixelGrids = new List<GridShape>();
        for(int i=0; i<_featureMaps.Count; i++)
        {
            pixelGrids.Add(_featureMaps[i].GetPixelGrid());
        }
        return pixelGrids;
    }

    override public void CalcMesh()
    {
        if (input == null)
        {
            base.CalcMesh();
            return;
        }

        _mesh.subMeshCount = 3;

        List<Vector3> verts = new List<Vector3>();
        List<Color> cols = new List<Color>();
        List<float> activations = new List<float>();
        List<int> inds = new List<int>();
        List<int> lineInds = new List<int>();
        List<int> polyInds = new List<int>();

        Vector3 posDiff = new Vector3(0, -yOffset, -zOffset);
        Vector3 zPos = CenterPosition();

        AddNodes(verts, inds);
        for (int i = 0; i < verts.Count; i++)
        {
            if (_activationTensorPerEpoch.ContainsKey(epoch))
            {

                int[] index = Util.GetSampleMultiDimIndices(_activationTensorShape, i, GlobalManager.Instance.testSample);

                Array activationTensor = _activationTensorPerEpoch[epoch];
                float tensorVal = (float)activationTensor.GetValue(index);

                if (allCalculations > 0)
                    activations.Add(tensorVal);

                tensorVal *= pointBrightness;

                cols.Add(new Color(tensorVal, tensorVal, tensorVal, 1f));

                //cols.Add(Color.white);
            }
            else
            {
                cols.Add(Color.black);
            }
        }


        //TODO: reuse generated vert positions

        //for each output feature map
        AddConvLines(verts, cols, activations, lineInds, posDiff, zPos);

        if (showOriginalDepth)
        {
            AllFeatureMapsDisplay allFeatureMapsDisplay = new AllFeatureMapsDisplay(new Vector3(0, fullResHeight, ZPosition()), fullDepth, lineCircleGrid, new Vector2(allFiltersSpacing, allFiltersSpacing));
            allFeatureMapsDisplay.AddPolysToLists(verts, polyInds);
            GridShape firstFeatureMapGrid = _featureMaps[_featureMaps.Count - 1].GetPixelGrid();
            allFeatureMapsDisplay.AddLinesToLists(verts, lineInds, firstFeatureMapGrid.BboxV(zPos.z));
        }


        //fill with dummy colors
        int diff = verts.Count - cols.Count;

        for (int i = 0; i < diff; i++)
        {
            cols.Add(Color.white);
        }

        _mesh.SetVertices(verts);
        _mesh.SetColors(cols);
        _mesh.SetIndices(inds.ToArray(), MeshTopology.Points, 0);
        _mesh.SetIndices(lineInds.ToArray(), MeshTopology.Lines, 1);
        _mesh.SetIndices(polyInds.ToArray(), MeshTopology.Triangles, 2);
    }

    protected virtual void AddConvLines(List<Vector3> verts, List<Color> cols, List<float> activations, List<int> lineInds, Vector3 posDiff, Vector3 zPos)
    {
        List<List<GridShape>> inputFeaturemapOutFilterList = _inputLayer.GetLineStartShapes(this, allCalculations, this.convLocation);

        for (int h = 0; h < _featureMaps.Count; h++)
        {
            //line endpoints
            GridShape outFeaturemapInputGrid = (GridShape)_featureMaps[h].GetGridForInputEndpoints(allCalculations);
            Vector3[] outFeaturemapVerts = outFeaturemapInputGrid.GetVertices(true);

            ProcessInputFeatureMaps(outFeaturemapVerts,
                inputFeaturemapOutFilterList,
                verts,
                cols, activations,
                lineInds, posDiff, zPos);
        }
    }

    protected void ProcessInputFeatureMaps(Vector3[] outFeaturemapVerts,
        List<List<GridShape>> inputFeaturemapNestedList,
        List<Vector3> verts,
        List<Color> cols,
        List<float> activations,
        List<int> lineInds,
        Vector3 posDiff,
        Vector3 zPos
)
    {
        foreach (List<GridShape> inputFeaturemapOutFilterList in inputFeaturemapNestedList)
        {
            ProcessInputFeatureMap(
                outFeaturemapVerts,
                inputFeaturemapOutFilterList,
                verts,
                cols,
                activations,
                lineInds,
                posDiff,
                zPos);
        }
    }

    protected void ProcessInputFeatureMap(
        Vector3[] outFeaturemapVerts,
        List<GridShape> inputFeaturemapOutFilterList,
        List<Vector3> verts, List<Color> cols,
        List<float> activations,
        List<int> lineInds,
        Vector3 posDiff,
        Vector3 zPos)
    {
        int convGridIndex = 0;
        foreach (GridShape convGrid in inputFeaturemapOutFilterList)
        {
            List<int> outFeaturemapVertInds = AddOutFeaturemapVertsForConvGrid(
                outFeaturemapVerts,
                convGridIndex,
                verts,
                cols,
                zPos);


            ProcessConvGrid(
                outFeaturemapVertInds,
                convGrid,
                verts,
                cols,
                activations,
                lineInds,
                posDiff,
                zPos);

            convGridIndex += 1;
        }
    }

    protected virtual List<int> AddOutFeaturemapVertsForConvGrid(
        Vector3[] allOutputFeaturemapVerts,
        int convGridIndex, List<Vector3> verts,
        List<Color> cols,
        Vector3 zPos)
    {
        List<int> outList = new List<int>();
        verts.Add(allOutputFeaturemapVerts[convGridIndex] + zPos);
        cols.Add(Color.black);
        outList.Add(verts.Count - 1);
        return outList;
    }

    protected void ProcessConvGrid(
        List<int> outFeaturemapVertInds,
        GridShape convGrid, List<Vector3> verts,
        List<Color> cols, List<float> activations,
        List<int> lineInds,
        Vector3 posDiff,
        Vector3 zPos)
    {
        foreach (Vector3 vert in convGrid.GetVertices(true))
        {
            verts.Add(vert + zPos + posDiff);
            cols.Add(Color.black);
            int endInd = verts.Count - 1;

            foreach (int startInd in outFeaturemapVertInds)
            {
                lineInds.Add(startInd);
                lineInds.Add(endInd);
            }
        }
    }

    protected virtual Vector3 EndVert(Vector3[] endverts, int inputFilterGrid)
    {
        return endverts[inputFilterGrid];
    }

    public override Vector3Int GetOutputShape()
    {
        return new Vector3Int(_featureMapResolution.x, _featureMapResolution.y, depth);
    }

    public override Vector2Int Get2DOutputShape()
    {
        return _featureMapResolution;
    }

    public override void SetExpansionLevel(float level)
    {
        if(level <= 1f)
        {
            edgeBundle = Mathf.Max(0, 1f - level);
            allCalculations = 0;
        }
        else if (level <= 2f)
        {
            edgeBundle = 0;
            allCalculations = 0;
        }
        else if(level > 2f && level <= 3f)
        {
            edgeBundle = 0;
            allCalculations = Mathf.Max(0, (level - 2f));
        }
    }

    public void SetWeightTensorForEpoch(Array tensor, int epoch)
    {
        _weightTensorPerEpoch[epoch] = tensor;
    }

    public Array GetWeightTensorForEpoch(int epoch)
    {
        return _weightTensorPerEpoch[epoch];
    }

    public void SetWeightTensorShape(int[] tensorShape)
    {
        this._weightTensorShape = tensorShape;
    }

    public int[] GetWeightTensorShape()
    {
        return _weightTensorShape;
    }

    public void SetActivationTensorForEpoch(Array tensor, int epoch)
    {
        _activationTensorPerEpoch.Add(epoch, tensor);
    }

    public Array GetActivationTensorForEpoch(int epoch)
    {
        return _activationTensorPerEpoch[epoch];
    }

    public void SetActivationTensorShape(int[] tensorShape)
    {
        this._activationTensorShape = tensorShape;
    }

    public int[] GetActivationTensorShape()
    {
        return _activationTensorShape;
    }

    public override bool Is2dLayer()
    {
        return true;
    }
}
