using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class MaxPoolLayer : InputAcceptingLayer, I2DMapLayer
{
    private Vector2Int _oldStride;

    private Vector2Int _featureMapResolution;
    private Vector2 _featureMapTheoreticalResolution;

    private Vector2Int _currentConvShape = new Vector2Int(1, 1);

    public float nodeSize;

    private List<FeatureMap> _featureMaps;

    private Dictionary<int, Array> _activationTensorPerEpoch = new Dictionary<int, Array>();
    private int[] _activationTensorShape = new int[4];

    /// <summary>
    /// Represents a Maxpool Layer of a CNN.
    /// </summary>
    public MaxPoolLayer()
    {
        zOffset = .1f;
    }

    /// <summary>
    /// Checks if parameters have been updated and reinitializes necessary data.
    /// </summary>
    protected override void UpdateForChangedParams()
    {
        base.UpdateForChangedParams();

        Vector2Int newRes = _featureMapResolution;
        int newDepth = reducedDepth;
        if (_inputLayer) {
            newRes = FeatureMap.GetFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));
            _featureMapTheoreticalResolution = FeatureMap.GetTheoreticalFloatFeatureMapShapeFromInput(_inputLayer.Get2DOutputShape(), convShape, stride, padding ? GetPadding() : new Vector2Int(0, 0));


            newDepth = _inputLayer.GetOutputShape().z;

            filterSpacing = ((ConvLayer)_inputLayer).filterSpacing;
            filterSpread = ((ConvLayer)_inputLayer).filterSpread;
            lineCircleGrid = ((ConvLayer)_inputLayer).lineCircleGrid;
        }

        if(newRes != _featureMapResolution ||
            newDepth != reducedDepth ||
            stride != _oldStride ||
            IsInitialized() == false ||
            _featureMaps == null)
        {
            _featureMapResolution = newRes;

            reducedDepth = newDepth;
            InitFeatureMaps();
            _oldStride = stride;
        }

        UpdateFeatureMaps();
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
        _currentConvShape = convShape;
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
        if(input == null)
        {
            base.CalcMesh();
            return;
        }

        _mesh.subMeshCount = 2;

        List<Vector3> verts = new List<Vector3>();
        List<Color> cols = new List<Color>();
        List<int> inds = new List<int>();

        Vector3 posDiff = new Vector3(0, 0, -zOffset);
        Vector3 zPos = new Vector3(0, 0, ZPosition());

        AddNodes(verts, inds);

        for (int i = 0; i < verts.Count; i++)
        {
            if (_activationTensorPerEpoch.ContainsKey(epoch))
            {

                int[] index = Util.GetSampleMultiDimIndices(_activationTensorShape, i, GlobalManager.Instance.testSample);

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

        List<List<Shape>> inputFilterPoints = _inputLayer.GetLineStartShapes(convShape, _featureMapResolution, _featureMapTheoreticalResolution, stride, allCalculations);

        List<int> lineInds = new List<int>();

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

            //for each input conv grid
            List<Shape> featureMapGrids = inputFilterPoints[h];
            for (int j = 0; j < featureMapGrids.Count; j++)
            {
                GridShape gridShape = (GridShape)featureMapGrids[j];
                Vector3[] start_verts = gridShape.GetVertices(true);

                if (j >= end_verts.Length)
                {
                    continue;
                }
                verts.Add(end_verts[j] + zPos);
                int start_ind = verts.Count - 1;
                int bundle_center_ind = 0;
                if (edgeBundle > 0)
                {
                    verts.Add(edgeBundleCenters[j]);
                    bundle_center_ind = verts.Count - 1;
                }
                for (int k = 0; k < start_verts.Length; k++)
                {
                    verts.Add(start_verts[k] + zPos + posDiff);

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


        _mesh.SetVertices(verts);
        int diff = verts.Count - cols.Count;
        for(int i=0; i< diff; i++)
        {
            cols.Add(Color.black);
        }
        _mesh.SetColors(cols);
        _mesh.SetIndices(inds.ToArray(), MeshTopology.Points, 0);
        _mesh.SetIndices(lineInds.ToArray(), MeshTopology.Lines, 1);
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
        if (level <= 1f)
        {
            edgeBundle = 1f-level;
            allCalculations = 0;
        }
        else if (level <= 2f)
        {
            edgeBundle = 0;
            allCalculations = 0;
        }
        else if (level > 2f && level <= 3f)
        {
            edgeBundle = 0;
            allCalculations = level - 2f;
        }
    }

    public void SetActivationTensorForEpoch(Array tensor, int epoch)
    {
        _activationTensorPerEpoch[epoch] = tensor;
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
