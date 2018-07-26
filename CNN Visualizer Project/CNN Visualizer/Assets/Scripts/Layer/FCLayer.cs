using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// Represents a Fully Connected Layer of a CNN.
/// </summary>
public class FCLayer : InputAcceptingLayer
{
    public int fullDepth = 1024;
    private int _oldFullDepth;

    private Vector3Int _inputResolution;

    [Range(0.0f, 1.0f)]
    public float collapseInput;

    public float nodeSize;

    private List<Vector3> _nodePositions;

    private Dictionary<int, Array> _tensorPerEpoch = new Dictionary<int, Array>();
    private int[] _tensorShape = new int[4];

    private Dictionary<int, Array> _activationTensorPerEpoch = new Dictionary<int, Array>();
    private int[] _activationTensorShape = new int[4];

    public FCLayer()
    {
        reducedDepth = 16;
    }

    /// <summary>
    /// Checks if parameters have been updated and reinitializes necessary data.
    /// </summary>
    protected override void UpdateForChangedParams()
    {
        base.UpdateForChangedParams();

        Vector3Int newRes = _inputResolution;
        if (_inputLayer) {
            newRes = _inputLayer.GetOutputShape();
        }

        if(reducedDepth != _oldReducedDepth ||
            newRes != _inputResolution ||
            IsInitialized() == false ||
            _nodePositions == null ||
            _oldFullDepth != fullDepth)
        {
            _inputResolution = newRes;
            InitNodes();
            _oldReducedDepth = reducedDepth;
            _oldFullDepth = fullDepth;
        }
    }

    /// <summary>
    /// Initialize data representing nodes of the layer.
    /// </summary>
    private void InitNodes()
    {
        Vector3[] filterPositions = GetInterpolatedNodePositions();

        _nodePositions = new List<Vector3>();
        for (int i = 0; i < reducedDepth; i++)
        {
            Vector3 node = filterPositions[i];
            _nodePositions.Add(node);
        }
    }

    /// <summary>
    /// Update data representing nodes of the layer.
    /// </summary>
    private void UpdateNodes()
    {
        Vector3[] filterPositions = GetInterpolatedNodePositions();

        for (int i=0; i<_nodePositions.Count; i++)
        {
            _nodePositions[i] = filterPositions[i];
        }
    }

    /// <summary>
    /// Returns a List of Shapes representing the pixels of the feature maps.
    /// </summary>
    /// <returns></returns>
    protected override List<Shape> GetPointShapes()
    {
        UpdateNodes();

        List<Shape> pixelGrids = new List<Shape>();
        for(int i=0; i<_nodePositions.Count; i++)
        {
            GridShape dummyGrid = new GridShape(_nodePositions[i], new Vector2Int(1, 1), new Vector2(0.0f, 0.0f));
            pixelGrids.Add(dummyGrid);
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
    public override List<List<Shape>> GetLineStartShapes(InputAcceptingLayer outputLayer, float allCalcs)
    {
        List<List<Shape>> list = new List<List<Shape>>();
        list.Add(GetPointShapes());
        return list;
    }


    override public void CalcMesh()
    {
        if(input == null)
        {
            base.CalcMesh();
            return;
        }

        _mesh.subMeshCount = 2;


        //POINTS
        List<Vector3> verts = new List<Vector3>();
        List<Color> cols = new List<Color>();
        List<float> activations = new List<float>();
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

                float tensorVal = (float)activationTensor.GetValue(index);

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



        if (showOriginalDepth)
        {
            AddFullResNodes(verts, inds);
        }
        int diff = verts.Count - cols.Count;
        for (int i = 0; i < diff; i++)
        {
            cols.Add(Color.black);
        }

        //LINES
        Vector2Int inputShape = new Vector2Int(1, 1);

        if(collapseInput != 1.0f)
        {
            inputShape = new Vector2Int(_inputLayer.GetOutputShape().x, _inputLayer.GetOutputShape().y);
        }

        //Input Layer Points, each inner List only contains one point
        List<List<Shape>> inputFilterPoints = _inputLayer.GetLineStartShapes(this, 0.0f); //TODO: Is this still correct after refactoring?

        List<int> lineInds = new List<int>();

        if (edgeBundle == 1f)
        {
            AddFullyBundledLines(verts, lineInds, inputFilterPoints);
            diff = verts.Count - cols.Count;
            for (int i = 0; i < diff; i++)
            {
                cols.Add(Color.black);
            }
        }
        else
        {

            //for each of this Layers Points
            for (int h = 0; h < _nodePositions.Count; h++)
            {
                //line endpoint
                Vector3 end_vert = _nodePositions[h];
                verts.Add(end_vert + zPos);
                cols.Add(Color.black);

                int end_ind = verts.Count - 1;

                Vector3 edgeBundleCenter = GetEdgeBundleCenter(end_vert, edgeBundle);
                int bundle_center_ind = 0;
                if (edgeBundle > 0)
                {

                    verts.Add(edgeBundleCenter);
                    cols.Add(Color.black);

                    bundle_center_ind = verts.Count - 1;
                }

                //for each input feature map
                for (int i = 0; i < inputFilterPoints.Count; i++)
                {
                    //for each input conv grid
                    List<Shape> featureMapGrids = inputFilterPoints[i];
                    for (int j = 0; j < featureMapGrids.Count; j++)
                    {
                        GridShape gridShape = (GridShape)featureMapGrids[j];
                        //scale shape spacing by collapse input
                        gridShape.spacing *= (1.0f - collapseInput);
                        Vector3[] start_verts = gridShape.GetVertices(true);

                        //for each conv point
                        for (int k = 0; k < start_verts.Length; k++)
                        {
                            lineInds.Add(end_ind);
                            if (edgeBundle > 0)
                            {
                                lineInds.Add(bundle_center_ind);
                                lineInds.Add(bundle_center_ind);
                            }

                            verts.Add(start_verts[k] + zPos + posDiff);
                            if (_tensorPerEpoch.ContainsKey(epoch))
                            {
                                Array tensor = _tensorPerEpoch[epoch];

                                int[] index = { i*k, h };
                                if (_inputLayer.GetType().Equals(typeof(FCLayer)))
                                {
                                    index[0] = j;
                                }

                                float activationMult = 1f;
                                if (GlobalManager.Instance.multWeightsByActivations)
                                    activationMult = activations[j];

                                float tensorVal = (float)tensor.GetValue(index) * weightBrightness * activationMult;
                                cols.Add(new Color(tensorVal, tensorVal, tensorVal, 1f));
                            }
                            else
                            {
                                cols.Add(Color.black);
                            }

                            lineInds.Add(verts.Count - 1);
                        }
                    }
                }
            }
        }


        _mesh.SetVertices(verts);
        _mesh.SetColors(cols);
        _mesh.SetIndices(inds.ToArray(), MeshTopology.Points, 0);
        _mesh.SetIndices(lineInds.ToArray(), MeshTopology.Lines, 1);

        Debug.Log("vertcount: " + verts.Count);
    }

    /// <summary>
    /// As we require much less vertices and line polygons when the edge bundling is turned up to 1.0, the mesh is calculated differently (in this function).
    /// </summary>
    /// <param name="verts"></param>
    /// <param name="lineInds"></param>
    /// <param name="inputFilterPoints"></param>
    void AddFullyBundledLines(List<Vector3> verts, List<int> lineInds, List<List<Shape>> inputFilterPoints)
    {
        Vector3 posDiff = new Vector3(0, 0, -zOffset);
        Vector3 zPos = new Vector3(0, 0, ZPosition());

        Vector3 end_vert0 = _nodePositions[0];
        Vector3 edgeBundleCenter = GetEdgeBundleCenter(end_vert0, 1f);
        verts.Add(edgeBundleCenter);
        int bundle_center_ind = verts.Count - 1;

        //for each of this Layers Points
        for (int h = 0; h < _nodePositions.Count; h++)
        {
            //line endpoint
            Vector3 end_vert = _nodePositions[h];
            verts.Add(end_vert + zPos);

            int end_ind = verts.Count - 1;

            lineInds.Add(end_ind);
            lineInds.Add(bundle_center_ind);
        }

        //for each input feature map
        for (int i = 0; i < inputFilterPoints.Count; i++)
        {
            //for each input conv grid
            List<Shape> featureMapGrids = inputFilterPoints[i];
            for (int j = 0; j < featureMapGrids.Count; j++)
            {
                GridShape gridShape = (GridShape)featureMapGrids[j];
                //scale shape spacing by collapse input
                gridShape.spacing *= (1.0f - collapseInput);
                Vector3[] start_verts = gridShape.GetVertices(true);

                //for each conv point
                for (int k = 0; k < start_verts.Length; k++)
                {

                    lineInds.Add(bundle_center_ind);

                    verts.Add(start_verts[k] + zPos + posDiff);
                    lineInds.Add(verts.Count - 1);
                }
            }
        }
    }

    /// <summary>
    /// Adds nodes visualizing the full, unreduced resolution of the layer.
    /// </summary>
    /// <param name="verts"></param>
    /// <param name="inds"></param>
    private void AddFullResNodes(List<Vector3> verts, List<int> inds)
    {
        Vector3[] vertsToAdd = null;

        if (lineCircleGrid < 1.0f)
        {
            Vector3[] lineVerts = fullResLineVerts();
            Vector3[] circleVerts = fullResCircleVerts();
            vertsToAdd = interpolateVectors(lineVerts, circleVerts, lineCircleGrid);
        }
        else if (lineCircleGrid <= 2.0f)
        {
            Vector3[] circleVerts = fullResCircleVerts();
            Vector3[] gridVerts = fullResGridVerts();
            vertsToAdd = interpolateVectors(circleVerts, gridVerts, lineCircleGrid - 1.0f);
        }

        for(int i=0; i<vertsToAdd.Length; i++)
        {
            verts.Add(vertsToAdd[i]);
            inds.Add(verts.Count - 1);
        }
    }

    /// <summary>
    /// helper to interpolate two vectors.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="alpha"></param>
    /// <returns></returns>
    private Vector3[] interpolateVectors(Vector3[] a, Vector3[] b, float alpha)
    {
        for(int i=0; i<Mathf.Min(a.Length, b.Length); i++)
        {
            a[i] = a[i] * (1.0f - alpha) + b[i] * alpha;
        }
        return a;
    }

    /// <summary>
    /// Returns the vertices of the line used for the full res visualization
    /// </summary>
    /// <returns></returns>
    private Vector3[] fullResLineVerts()
    {
        int reducedDepthGridShape = Mathf.CeilToInt(Mathf.Sqrt(reducedDepth));

        LineShape line = new LineShape(CenterPosition(), fullDepth, filterSpread * (reducedDepthGridShape / (float)reducedDepth));
        return line.GetVertices(false);
    }

    /// <summary>
    /// Returns the vertices of the circles used for the full res visualization
    /// </summary>
    /// <returns></returns>
    private Vector3[] fullResCircleVerts()
    {
        List<Vector3> outVerts = new List<Vector3>();
        int circleNum = Mathf.CeilToInt( fullDepth / (float)reducedDepth);

        for(int i=0; i<circleNum; i++)
        {
            CircleShape circle = new CircleShape(CenterPosition(), reducedDepth, filterSpread * 1.3f + i*2.0f*(filterSpread / 10.0f));
            outVerts.AddRange(circle.GetVertices(false));
        }


        return outVerts.ToArray();
    }

    /// <summary>
    /// Returns the vertices of the grid used for the full res visualization
    /// </summary>
    /// <returns></returns>
    private Vector3[] fullResGridVerts()
    {
        int root = Mathf.CeilToInt(Mathf.Sqrt(fullDepth));

        GridShape grid = new GridShape(CenterPosition(), new Vector2Int(root, root), new Vector2(filterSpread, filterSpread));
        return grid.GetVertices(false);
    }

    public override Vector3Int GetOutputShape()
    {
        return new Vector3Int(1, 1, reducedDepth);
    }

    public override void SetExpansionLevel(float level)
    {
        if (level <= 1f)
        {
            edgeBundle = 1f - level;
            collapseInput = 1f;
        }
        else if (level <= 2f)
        {
            edgeBundle = 0;
            collapseInput = 1f - (level - 1f);
        } else
        {
            edgeBundle = 0;
            collapseInput = 0;
        }
    }

    public void SetTensorForEpoch(Array tensor, int epoch)
    {
        _tensorPerEpoch[epoch] = tensor;
    }

    public Array GetTensorForEpoch(int epoch)
    {
        return _tensorPerEpoch[epoch];
    }

    public void SetTensorShape(int[] tensorShape)
    {
        this._tensorShape = tensorShape;
    }

    public int[] GetTensorShape()
    {
        return _tensorShape;
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
}
