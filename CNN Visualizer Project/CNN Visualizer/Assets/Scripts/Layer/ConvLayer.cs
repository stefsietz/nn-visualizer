using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// Represents a Convolutional Layer of a CNN.
/// </summary>
public class ConvLayer : InputAcceptingLayer, I2DMapLayer
{
    private Vector2Int oldReducedShape;
    private Vector2Int oldStride;

    public int fullDepth = 64;
    private int oldFullDepth;

    private bool oldPadding;

    private Vector2Int featureMapResolution;
    private Vector2 featureMapTheoreticalResolution;

    private Vector2Int currentConvShape = new Vector2Int(1, 1);

    [Range(0.0f, 1.0f)]
    public float allFiltersSpacing = 0.5f;

    public float fullResHeight = 5.0f;

    public float nodeSize;

    private List<FeatureMap> _featureMaps;

    private Dictionary<int, Array> _weightTensorPerEpoch = new Dictionary<int, Array>();
    private int[] weightTensorShape = new int[4];

    private Dictionary<int, Array> _activationTensorPerEpoch = new Dictionary<int, Array>();
    private int[] activationTensorShape = new int[4];

    public ConvLayer()
    {
        reducedShape = new Vector2Int(3, 3);
        oldPadding = padding;
    }

    /// <summary>
    /// Checks if parameters have been updated and reinitializes necessary data.
    /// </summary>
    protected override void UpdateForChangedParams()
    {
        base.UpdateForChangedParams();

        Vector2Int newRes = featureMapResolution;
        if (_inputLayer) {
            newRes = FeatureMap.GetFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), reducedShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
            featureMapTheoreticalResolution = FeatureMap.GetTheoreticalFloatFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), reducedShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
        }

        if(reducedDepth != oldReducedDepth ||
            reducedShape != oldReducedShape ||
            newRes != featureMapResolution ||
            stride != oldStride ||
            padding != oldPadding ||
            IsInitialized() == false ||
            _featureMaps == null)
        {
            featureMapResolution = newRes;

            InitFeatureMaps();
            oldReducedShape = reducedShape;
            oldReducedDepth = reducedDepth;
            oldStride = stride;
            oldPadding = padding;
        }
    }

    /// <summary>
    /// Initializes feature map List.
    /// </summary>
    private void InitFeatureMaps()
    {
        Vector3[] filterPositions = GetInterpolatedNodePositions();

        _featureMaps = new List<FeatureMap>();
        for (int i = 0; i < reducedDepth; i++)
        {
            FeatureMap map = new FeatureMap(this, i); 
            _featureMaps.Add(map);
        }

    }

    /// <summary>
    /// Updates feature map list according to new parameters.
    /// </summary>
    private void UpdateFeatureMaps()
    {
        if (_featureMaps == null)
        {
            InitFeatureMaps();
            return;
        }

        Vector3[] filterPositions = GetInterpolatedNodePositions();

        for (int i=0; i<_featureMaps.Count; i++)
        {
            _featureMaps[i].UpdateValues(this);
        }
    }

    /// <summary>
    /// Returns a List of Shapes representing the pixels of the feature maps.
    /// </summary>
    /// <returns></returns>
    protected override List<Shape> GetPointShapes()
    {
        UpdateFeatureMaps();

        List<Shape> pixelGrids = new List<Shape>();
        for(int i=0; i<_featureMaps.Count; i++)
        {
            pixelGrids.Add(_featureMaps[i].GetPixelGrid());
        }
        return pixelGrids;
    }

    /// <summary>
    /// Calculates and returns the positions of the line start points (for the CalcMesh() function)
    /// </summary>
    /// <param name="convShape"></param>
    /// <param name="outputShape"></param>
    /// <param name="theoreticalOutputShape"></param>
    /// <param name="stride"></param>
    /// <param name="allCalcs"></param>
    /// <returns></returns>
    public override List<List<Shape>> GetLineStartShapes(Vector2Int convShape, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs)
    {
        currentConvShape = convShape;
        UpdateFeatureMaps();

        List<List<Shape>> filterGrids = new List<List<Shape>>();
        for (int i = 0; i < _featureMaps.Count; i++)
        {
            filterGrids.Add(_featureMaps[i].GetFilterGrids(outputShape, theoreticalOutputShape, stride, allCalcs));
        }
        return filterGrids;
    }

    override public void CalcMesh()
    {
        Debug.Log(name);
        if(_input == null)
        {
            base.CalcMesh();
            return;
        }

        _mesh.subMeshCount = 3;

        List<Vector3> verts = new List<Vector3>();
        List<Color> cols = new List<Color>();
        List<int> inds = new List<int>();
        List<int> lineInds = new List<int>();
        List<int> polyInds = new List<int>();

        Vector3 posDiff = new Vector3(0, 0, -zOffset);
        Vector3 zPos = new Vector3(0, 0, ZPosition());

        AddNodes(verts, inds);
        for (int i = 0; i < verts.Count; i++)
        {
            if (_activationTensorPerEpoch.ContainsKey(epoch))
            {

                int[] index = Util.GetMultiDimIndices(activationTensorShape, i);

                Array activationTensor = _activationTensorPerEpoch[epoch];
                float tensorVal = (float)activationTensor.GetValue(index) * pointBrightness;
                cols.Add(new Color(tensorVal, tensorVal, tensorVal, 1f));

                //cols.Add(Color.white);
            }
            else
            {
                cols.Add(Color.black);
            }
        }


        List<List<Shape>> inputFilterPoints = _inputLayer.GetLineStartShapes(reducedShape, featureMapResolution, featureMapTheoreticalResolution, stride, allCalculations);

        //TODO: reuse generated vert positions

        //for each output feature map
        for (int h = 0; h < _featureMaps.Count; h++)
        {
            //line endpoints
            GridShape inputGrid = (GridShape)_featureMaps[h].GetInputGrid(allCalculations);
            Vector3[] end_verts = inputGrid.GetVertices(true);

            Vector3[] edgeBundleCenters = new Vector3[end_verts.Length];

            for(int i = 0; i<edgeBundleCenters.Length; i++)
            {
                edgeBundleCenters[i] = GetEdgeBundleCenter(end_verts[i], edgeBundle);
            }


            //for each input feature map
            for (int i = 0; i < inputFilterPoints.Count; i++)
            {
                //for each input conv grid
                List<Shape> featureMapGrids = inputFilterPoints[i];
                for (int j = 0; j < featureMapGrids.Count; j++)
                {
                    GridShape gridShape = (GridShape)featureMapGrids[j];
                    Vector3[] start_verts = gridShape.GetVertices(true);

                    verts.Add(end_verts[j] + zPos);
                    cols.Add(Color.black);
                    int start_ind = verts.Count - 1;
                    int bundle_center_ind = 0;
                    if (edgeBundle > 0)
                    {
                        verts.Add(edgeBundleCenters[j]);
                        cols.Add(Color.black);
                        bundle_center_ind = verts.Count - 1;
                    }
                    for (int k = 0; k < start_verts.Length; k++)
                    {
                        verts.Add(start_verts[k] + zPos + posDiff);

                        if (_weightTensorPerEpoch.ContainsKey(epoch)){
                            Array tensor = _weightTensorPerEpoch[epoch];
                            int kernelInd1 = k % reducedShape.x;
                            int kernelInd2 = k / reducedShape.y;

                            int[] index = { kernelInd1, kernelInd2, 0, h };

                            float tensorVal = (float)tensor.GetValue(index) * weightBrightness;
                            cols.Add(new Color(tensorVal, tensorVal, tensorVal, 1f));
                        } else
                        {
                            cols.Add(Color.black);
                        }

                        lineInds.Add(start_ind);
                        if (edgeBundle > 0)
                        {
                            lineInds.Add(bundle_center_ind);
                            lineInds.Add(bundle_center_ind);
                        }
                        lineInds.Add(verts.Count - 1);
                    }
                }
            }
        }

        if (showOriginalDepth)
        {
            AllFeatureMapsDisplay allFeatureMapsDisplay = new AllFeatureMapsDisplay(new Vector3(0, fullResHeight, ZPosition()), fullDepth, lineCircleGrid, new Vector2(allFiltersSpacing, allFiltersSpacing));
            allFeatureMapsDisplay.AddPolysToLists(verts, polyInds);
            GridShape firstFeatureMapGrid = _featureMaps[_featureMaps.Count - 1].GetPixelGrid();
            allFeatureMapsDisplay.AddLinesToLists(verts, lineInds, firstFeatureMapGrid.BboxV(zPos.z));
        }


        //fill with dummy colors
        int diff = verts.Count - cols.Count;
        Debug.Log("diff " + diff+ " polyInds " + polyInds.Count + " inds " + inds.Count + " lineInds " + lineInds.Count  + " cols " + cols.Count + " verts " + verts.Count);
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

    public override Vector3Int GetOutputShape()
    {
        return new Vector3Int(featureMapResolution.x, featureMapResolution.y, reducedDepth);
    }

    public override Vector2Int Get2DOutputShape()
    {
        return featureMapResolution;
    }

    public override void SetExpansionLevel(float level)
    {
        if(level <= 1f)
        {
            edgeBundle = Mathf.Max(0, 1f - level);
            allCalculations = 0;
        } else if(level <= 2f)
        {
            edgeBundle = 0;
            allCalculations = Mathf.Max(0, (level - 1f));
        }
    }

    public void SetWeightTensorForEpoch(Array tensor, int epoch)
    {
        _weightTensorPerEpoch.Add(epoch, tensor);
    }

    public Array GetWeightTensorForEpoch(int epoch)
    {
        return _weightTensorPerEpoch[epoch];
    }

    public void SetWeightTensorShape(int[] tensorShape)
    {
        this.weightTensorShape = tensorShape;
    }

    public int[] GetWeightTensorShape()
    {
        return weightTensorShape;
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
        this.activationTensorShape = tensorShape;
    }

    public int[] GetActivationTensorShape()
    {
        return activationTensorShape;
    }

    public FeatureMapInfo GetFeatureMapInfo(int featureMapIndex)
    {
        Vector3[] filterPositions = GetInterpolatedNodePositions(); //not ideal  recalculating this everytime, but should have minor performance impact
        FeatureMapInfo info = new FeatureMapInfo();
        info.position = filterPositions[featureMapIndex];
        info.shape = featureMapResolution;
        info.filterShape = currentConvShape;
        info.outputShape = Get2DOutputShape();
        info.spacing = filterSpacing;
        return info;
    }
}
