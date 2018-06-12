using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Class for handling the display of the full number of feature maps as squares above its convolutional layer.
/// </summary>
public class AllFeatureMapsDisplay
{
    public Vector3 position;
    public Vector2 spacing;

    public float lineToGrid;

    public int filterNum;

    public AllFeatureMapsDisplay(Vector3 position, int filterNum, float lineToGrid, Vector2 spacing)
    {
        this.position = position;
        this.filterNum = filterNum;
        this.lineToGrid = lineToGrid;
        this.spacing = spacing;
    }

    public void Update(Vector3 position, int filterNum, float lineToGrid, Vector2 spacing)
    {
        this.position = position;
        this.filterNum = filterNum;
        this.lineToGrid = lineToGrid;
        this.spacing = spacing;
    }

    private Vector3[] GetInterpolatedFilterPositions()
    {
        int filterNumGridShape = Mathf.CeilToInt(Mathf.Sqrt(filterNum));
        Vector3[] gridPositions = GridShape.ScaledUnitGrid(filterNumGridShape, filterNumGridShape, Vector3.zero, spacing.x);
        Vector3[] linePositions = LineShape.ScaledUnitLine(filterNum, Vector3.zero, new Vector3(1, 0, 0), spacing.x);

        Vector3[] filterPositions = Shape.InterpolateShapes(linePositions, gridPositions, 1.0f); // TOFO: Maybe integrate lineToGrid parameter, but for now only grid
        return filterPositions;
    }

    public void AddPolysToLists(List<Vector3> verts, List<int> inds)
    {
        Vector3[] filterPositions = GetInterpolatedFilterPositions();

        for(int i=0; i<filterPositions.Length; i++)
        {
            Vector3 pos = position + filterPositions[i];
            int baseInd = verts.Count;


            //reverse x pos to get first element into pos x axis

            Vector3[] square = SquareAtPosition(pos, spacing);

            verts.AddRange(square);

            inds.Add(baseInd + 0);
            inds.Add(baseInd + 1);
            inds.Add(baseInd + 2);
            inds.Add(baseInd + 0);
            inds.Add(baseInd + 2);
            inds.Add(baseInd + 3);
        }
    }

    public void AddLinesToLists(List<Vector3> verts, List<int> inds, Vector3[] startPoints)
    {
        Vector3[] filterPositions = GetInterpolatedFilterPositions();

        Vector3 pos = position + filterPositions[0];
        int baseInd = verts.Count;

        Vector3[] endPoints = SquareAtPosition(pos, spacing);

        verts.AddRange(startPoints);
        verts.AddRange(endPoints);

        inds.Add(baseInd + 0);
        inds.Add(baseInd + 6);
        inds.Add(baseInd + 1);
        inds.Add(baseInd + 7);
        inds.Add(baseInd + 2);
        inds.Add(baseInd + 4);
        inds.Add(baseInd + 3);
        inds.Add(baseInd + 5);
    }

    private Vector3[] SquareAtPosition(Vector3 pos, Vector2 spacing)
    {
        Vector3 p1 = new Vector3(-pos.x + spacing.x / 4, pos.y + spacing.y / 4, pos.z);
        Vector3 p2 = new Vector3(-pos.x - spacing.x / 4, pos.y + spacing.y / 4, pos.z);
        Vector3 p3 = new Vector3(-pos.x - spacing.x / 4, pos.y - spacing.y / 4, pos.z);
        Vector3 p4 = new Vector3(-pos.x + spacing.x / 4, pos.y - spacing.y / 4, pos.z);

        Vector3[] o = new Vector3[4];
        o[0] = p1;
        o[1] = p2;
        o[2] = p3;
        o[3] = p4;

        return o;
    }

}
