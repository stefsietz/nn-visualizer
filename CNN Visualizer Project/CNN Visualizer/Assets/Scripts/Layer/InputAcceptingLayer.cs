using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using System.Collections;

public abstract class InputAcceptingLayer : Layer
{
    private GameObject _oldInput;
    public GameObject input;

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

    /// <summary>
    /// feature map always refers to the layers' output
    /// </summary>
    protected Vector2Int _featureMapResolution;
    /// <summary>
    /// feature map always refers to the layers' output
    /// </summary>
    protected Vector2 _featureMapTheoreticalResolution;

    public abstract void SetExpansionLevel(float level);
    public abstract bool Is2dLayer();

    protected override void UpdateForChangedParams(bool topoChanged)
    {
        base.UpdateForChangedParams(topoChanged);

        if (!HasInputLayer())
            return;

        _inputLayer.UpdateFeatureMapsForOutputParams(this, topoChanged);
    }

    private bool HasInputLayer()
    {
        return (_inputLayer != null);
    }

    protected override void CheckAndHandleParamChanges()
    {

        if (CheckInputChanged_OneValidCall())
        {
            InputChanged();
        }

        base.CheckAndHandleParamChanges();
    }

    public override void InitIfUnitialized()
    {
        if (_initialized)
            return;

        base.InitIfUnitialized();

        _initialized = true;
    }

    protected void InputChanged()
    {
        CheckAndHandleInputChange();

        UpdateMeshTopoChanged();
    }

    protected void CheckAndHandleInputChange()
    {
        if (input != null)
        {

            Layer newInputLayer = input.GetComponent<Layer>();
            if (!newInputLayer.HasInitializedMesh())
            {
                newInputLayer.InitMesh();
            }

            if (_inputLayer != null)
            {
                _inputLayer.RemoveObserver(this);
                _inputLayer.RemoveOutputLayer(this);
            }

            newInputLayer.AddObserver(this);
            newInputLayer.AddOuputLayer(this);
            _inputLayer = newInputLayer;

        }
        else
        {
            _inputLayer = null;
        }
    }

    /// <summary>
    /// Calculates the position of the Feature Maps according to parameters.
    /// </summary>
    /// <returns></returns>
    protected override Vector3[] GetInterpolatedFeatureMapPositions()
    {
        int reducedDepthGridShape = Mathf.CeilToInt(Mathf.Sqrt(depth));
        Vector3[] gridPositions = GridShape.ScaledUnitGrid(reducedDepthGridShape, reducedDepthGridShape, new Vector3(0, 0, 0), filterSpread);
        Vector3[] linePositionsX = LineShape.ScaledUnitLine(depth, new Vector3(0, 0, 0), new Vector3(1, 0, 0), filterSpread * (reducedDepthGridShape / (float)depth));
        Vector3[] linePositionsZ = LineShape.ScaledUnitLine(depth, new Vector3(0, 0, 0), new Vector3(0, 0, 1), filterSpread * (reducedDepthGridShape / (float)depth));
        Vector3[] linePositions = Shape.InterpolateShapes(linePositionsX, linePositionsZ, lineXToZ);
        Vector3[] circlePositions = CircleShape.ScaledUnitCircle(depth, new Vector3(0, 0, 0), filterSpread);

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

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckInputChanged_OneValidCall()
    {
        if (input != _oldInput)
        {
            _oldInput = input;
            return true;
        }
        return false;
    }

    public override FeatureMapInputProperties GetFeatureMapInputProperties(int featureMapIndex)
    {
        Vector3[] filterPositions = GetInterpolatedFeatureMapPositions(); //not ideal  recalculating this everytime, but should have minor performance impact
        FeatureMapInputProperties info = new FeatureMapInputProperties();
        info.position = filterPositions[featureMapIndex];
        info.inputShape = _featureMapResolution;
        info.spacing = filterSpacing;
        return info;
    }

}
