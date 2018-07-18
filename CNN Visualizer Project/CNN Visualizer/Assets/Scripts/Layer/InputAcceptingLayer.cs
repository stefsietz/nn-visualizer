using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using System.Collections;

public abstract class InputAcceptingLayer : Layer
{
    public GameObject input;

    public int reducedDepth = 4;
    protected int _oldReducedDepth;

    [Range(0.0f, 1.0f)]
    public float filterSpread;

    [Range(0.0f, 0.1f)]
    public float filterSpacing;

    [Range(0.0f, 1.0f)]
    public float edgeBundle;

    [Range(0.0f, 1.0f)]
    public float allCalculations;

    [Range(0.0f, 2.0f)]
    public float lineCircleGrid;

    [Range(0.0f, 1.0f)]
    public float lineXToZ;

    public abstract void SetExpansionLevel(float level);

    protected override void UpdateForChangedParams()
    {
        CheckAndHandleInputChange();
        base.UpdateForChangedParams();
    }

    protected void CheckAndHandleInputChange()
    {
        if (input != null)
        {
            Layer newInputLayer = input.GetComponent<Layer>();
            if (!newInputLayer.IsInitialized())
            {
                newInputLayer.Init();
            }
            if (_inputLayer != newInputLayer)
            {
                if (_inputLayer != null)
                {
                    _inputLayer.RemoveObserver(this);
                }
                newInputLayer.AddObserver(this);
                _inputLayer = newInputLayer;
            }
        }
        else
        {
            if (_inputLayer != null)
                _inputLayer.RemoveObserver(this);
            _inputLayer = null;
        }
    }

    /// <summary>
    /// Calculates the position of the nodes according to parameters.
    /// </summary>
    /// <returns></returns>
    protected Vector3[] GetInterpolatedNodePositions()
    {
        int reducedDepthGridShape = Mathf.CeilToInt(Mathf.Sqrt(reducedDepth));
        Vector3[] gridPositions = GridShape.ScaledUnitGrid(reducedDepthGridShape, reducedDepthGridShape, new Vector3(0, 0, 0), filterSpread);
        Vector3[] linePositionsX = LineShape.ScaledUnitLine(reducedDepth, new Vector3(0, 0, 0), new Vector3(1, 0, 0), filterSpread * (reducedDepthGridShape / (float)reducedDepth));
        Vector3[] linePositionsZ = LineShape.ScaledUnitLine(reducedDepth, new Vector3(0, 0, 0), new Vector3(0, 0, 1), filterSpread * (reducedDepthGridShape / (float)reducedDepth));
        Vector3[] linePositions = Shape.InterpolateShapes(linePositionsX, linePositionsZ, lineXToZ);
        Vector3[] circlePositions = CircleShape.ScaledUnitCircle(reducedDepth, new Vector3(0, 0, 0), filterSpread);

        Vector3[] filterPositions = null;

        if (lineCircleGrid < 1.0f)
        {
            filterPositions = Shape.InterpolateShapes(linePositions, circlePositions, lineCircleGrid);
        }
        else if (lineCircleGrid <= 2.0f)
        {
            filterPositions = Shape.InterpolateShapes(circlePositions, gridPositions, lineCircleGrid - 1.0f);
        }
        return filterPositions;
    }

    /// <summary>
    /// Returns the edgebundle position for the corresponding point.
    /// </summary>
    /// <param name="originalPoint"></param>
    /// <param name="edgeBundle"></param>
    /// <returns></returns>
    protected Vector3 GetEdgeBundleCenter(Vector3 originalPoint, float edgeBundle)
    {
        float zPos = ZPosition();
        Vector3 center = new Vector3(0, 0, zPos - zOffset / 2.0f);
        originalPoint.z = zPos;
        return center = edgeBundle * center + (1.0f - edgeBundle) * originalPoint;
    }
}
