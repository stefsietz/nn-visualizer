using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

/// <summary>
/// Represents a Fully Connected Layer of a CNN.
/// </summary>
public class FCLayer : ConvLayer
{
    [Range(0.0f, 1.0f)]
    public float collapseInput;

    public FCLayer()
    {
        depth = 16;
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

    public override List<List<GridShape>> GetLineStartShapes(InputAcceptingLayer outputLayer, float allCalcs, int convLocation)
    {
        List<List<GridShape>> list = new List<List<GridShape>>();
        list.Add(GetPointShapes());
        return list;
    }

    protected override void AddConvLines(List<Vector3> verts, List<Color> cols, List<float> activations, List<int> lineInds, Vector3 posDiff, Vector3 zPos)
    {
        List<List<GridShape>> inputFeaturemapOutFilterList = _inputLayer.GetLineStartShapes(this, allCalculations, this.convLocation);

        for (int h = 0; h < _featureMaps.Count; h++)
        {
            //line endpoints
            Vector3[] outFeaturemapVerts = GetFeatureMapPositions().ToArray();

            ProcessInputFeatureMaps(outFeaturemapVerts,
                inputFeaturemapOutFilterList,
                verts,
                cols, activations,
                lineInds, posDiff, zPos);
        }
    }

    protected override List<int> AddOutFeaturemapVertsForConvGrid(
        Vector3[] allOutputFeaturemapVerts,
        int convGridIndex, List<Vector3> verts,
        List<Color> cols,
        Vector3 zPos)
    {
        List<int> outList = new List<int>();
        foreach (Vector3 vert in allOutputFeaturemapVerts)
        {
            verts.Add(vert + zPos);
            cols.Add(Color.black);
            outList.Add(verts.Count - 1);
        }
        return outList;
    }

    protected override Vector3 EndVert(Vector3[] endverts, int inputFilterGrid)
    {
        return endverts[0];
    }

    /// <summary>
    /// As we require much less vertices and line polygons when the edge bundling is turned up to 1.0, the mesh is calculated differently (in this function).
    /// </summary>
    /// <param name="verts"></param>
    /// <param name="lineInds"></param>
    /// <param name="inputFilterPoints"></param>
    void AddFullyBundledLines(List<Vector3> verts, List<int> lineInds, List<List<GridShape>> inputFilterPoints)
    {
        Vector3 posDiff = new Vector3(0, 0, -zOffset);
        Vector3 zPos = new Vector3(0, 0, ZPosition());

        List<Vector3> featureMapPositions = GetFeatureMapPositions();

        Vector3 end_vert0 = featureMapPositions[0];
        Vector3 edgeBundleCenter = GetEdgeBundleCenter(end_vert0, 1f);
        verts.Add(edgeBundleCenter);
        int bundle_center_ind = verts.Count - 1;

        //for each of this Layers Points
        for (int h = 0; h < featureMapPositions.Count; h++)
        {
            //line endpoint
            Vector3 end_vert = featureMapPositions[h];
            verts.Add(end_vert + zPos);

            int end_ind = verts.Count - 1;

            lineInds.Add(end_ind);
            lineInds.Add(bundle_center_ind);
        }

        //for each input feature map
        for (int i = 0; i < inputFilterPoints.Count; i++)
        {
            //for each input conv grid
            List<GridShape> featureMapGrids = inputFilterPoints[i];
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


    private List<Vector3> GetFeatureMapPositions()
    {
        List<Vector3> featureMapPositions = new List<Vector3>();

        foreach(FeatureMap map in _featureMaps)
        {
            featureMapPositions.Add(map.GetPosition());
        }

        return featureMapPositions;
    }

    protected override void SetupFeaturemapResolution()
    {
        _featureMapResolution = new Vector2Int(1, 1);
        _featureMapTheoreticalResolution = new Vector2Int(1, 1);
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
        int reducedDepthGridShape = Mathf.CeilToInt(Mathf.Sqrt(depth));

        LineShape line = new LineShape(CenterPosition(), fullDepth, filterSpread * (reducedDepthGridShape / (float)depth));
        return line.GetVertices(false);
    }

    /// <summary>
    /// Returns the vertices of the circles used for the full res visualization
    /// </summary>
    /// <returns></returns>
    private Vector3[] fullResCircleVerts()
    {
        List<Vector3> outVerts = new List<Vector3>();
        int circleNum = Mathf.CeilToInt( fullDepth / (float)depth);

        for(int i=0; i<circleNum; i++)
        {
            CircleShape circle = new CircleShape(CenterPosition(), depth, filterSpread * 1.3f + i*2.0f*(filterSpread / 10.0f));
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
        return new Vector3Int(1, 1, depth);
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

    public override bool Is2dLayer()
    {
        return false;
    }
}
