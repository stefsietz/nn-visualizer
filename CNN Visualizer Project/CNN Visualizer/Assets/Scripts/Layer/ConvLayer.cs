using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// Represents a Convolutional Layer of a CNN.
/// </summary>
public class ConvLayer : InputAcceptingLayer, I2DMapLayer
{
    private Vector2Int _oldConvShape;
    private Vector2Int _oldStride;

    public int fullDepth = 64;
    private int _oldFullDepth;

    private bool _oldPadding;

    /// <summary>
    /// feature map always refers to the layers' output
    /// </summary>
    private Vector2Int _featureMapResolution;
    /// <summary>
    /// feature map always refers to the layers' output
    /// </summary>
    private Vector2 _featureMapTheoreticalResolution;

    private Vector2Int _currentConvShape = new Vector2Int(1, 1);

    [Range(0.0f, 1.0f)]
    public float allFiltersSpacing = 0.5f;

    [Range(-1, 128)]
    public int convLocation = 0;

    public float fullResHeight = 5.0f;

    public float nodeSize;

    private List<FeatureMap> _featureMaps;

    private Dictionary<int, Array> _weightTensorPerEpoch = new Dictionary<int, Array>();
    private int[] _weightTensorShape = new int[4];

    private Dictionary<int, Array> _activationTensorPerEpoch = new Dictionary<int, Array>();
    private int[] _activationTensorShape = new int[4];

    public ConvLayer()
    {
        convShape = new Vector2Int(3, 3);
        _oldPadding = padding;
    }

    /// <summary>
    /// Checks if parameters have been updated and reinitializes necessary data.
    /// </summary>
    protected override void UpdateForChangedParams()
    {
        base.UpdateForChangedParams();

        Vector2Int newRes = _featureMapResolution;
        if (_inputLayer) {
            newRes = FeatureMap.GetFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
            _featureMapTheoreticalResolution = FeatureMap.GetTheoreticalFloatFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
        }

        if(reducedDepth != _oldReducedDepth ||
            convShape != _oldConvShape ||
            newRes != _featureMapResolution ||
            stride != _oldStride ||
            padding != _oldPadding ||
            IsInitialized() == false ||
            _featureMaps == null)
        {
            _featureMapResolution = newRes;

            InitFeatureMaps();
            _oldConvShape = convShape;
            _oldReducedDepth = reducedDepth;
            _oldStride = stride;
            _oldPadding = padding;
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
    /// Calculates and returns the positions of the line start points (for the CalcMesh() function). Gets Called by a Layer that is connected to this Layers output.
    /// </summary>
    /// <param name="convShape">Shape of the Conv operation of the next Layer</param>
    /// <param name="outputShape">Output Shape</param>
    /// <param name="theoreticalOutputShape"></param>
    /// <param name="stride"></param>
    /// <param name="allCalcs"></param>
    /// <returns></returns>
    public override List<List<Shape>> GetLineStartShapes(Vector2Int convShape, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs)
    {
        _currentConvShape = convShape;
        UpdateFeatureMaps();

        List<List<Shape>> filterGrids = new List<List<Shape>>();
        for (int i = 0; i < _featureMaps.Count; i++)
        {
            filterGrids.Add(_featureMaps[i].GetFilterGrids(outputShape, theoreticalOutputShape, stride, allCalcs));
        }
        return filterGrids;
    }

    public override List<List<Shape>> GetLineStartShapes(Vector2Int convShape, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs, int convLocation)
    {
        _currentConvShape = convShape;
        UpdateFeatureMaps();

        List<List<Shape>> filterGrids = new List<List<Shape>>();
        for (int i = 0; i < _featureMaps.Count; i++)
        {
            filterGrids.Add(_featureMaps[i].GetFilterGrids(outputShape, theoreticalOutputShape, stride, allCalcs, convLocation));
        }
        return filterGrids;
    }

    override public void CalcMesh()
    {
        Debug.Log(name);
        if(input == null)
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

        Vector3 posDiff = new Vector3(0, 0, -zOffset);
        Vector3 zPos = new Vector3(0, 0, ZPosition());

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


        List<List<Shape>> inputFilterPoints = _inputLayer.GetLineStartShapes(convShape, _featureMapResolution, _featureMapTheoreticalResolution, stride, allCalculations);
        inputFilterPoints = _inputLayer.GetLineStartShapes(convShape, _featureMapResolution, _featureMapTheoreticalResolution, stride, allCalculations, this.convLocation);

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
                            int kernelInd1 = k % convShape.x;
                            int kernelInd2 = k / convShape.y;

                            int[] index = { kernelInd1, kernelInd2, 0, h };

                            float activationMult = 1f;
                            if (allCalculations > 0  && GlobalManager.Instance.multWeightsByActivations)
                                activationMult = activations[j];

                            float tensorVal = (float)tensor.GetValue(index) * weightBrightness * activationMult;
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
        return new Vector3Int(_featureMapResolution.x, _featureMapResolution.y, reducedDepth);
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

    public FeatureMapInfo GetFeatureMapInfo(int featureMapIndex)
    {
        Vector3[] filterPositions = GetInterpolatedNodePositions(); //not ideal  recalculating this everytime, but should have minor performance impact
        FeatureMapInfo info = new FeatureMapInfo();
        info.position = filterPositions[featureMapIndex];
        info.shape = _featureMapResolution;
        info.convShape = _currentConvShape;
        info.outputShape = Get2DOutputShape();
        info.spacing = filterSpacing;
        return info;
    }
}
