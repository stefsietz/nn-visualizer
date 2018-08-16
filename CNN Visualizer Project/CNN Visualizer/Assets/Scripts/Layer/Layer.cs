using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

// Runtime code here
#if UNITY_EDITOR
[ExecuteInEditMode]
#endif
// Runtime code here
[RequireComponent(typeof(MeshFilter))]

/// Base class for the layer component
public abstract class Layer : MonoBehaviour, I2DMapLayer
{
    //TODO: these should be more class specific
    protected Vector2Int _oldConvShape;
    public Vector2Int convShape;

    public int depth = 4;
    protected int _oldDepth;

    protected Vector2Int _oldStride = new Vector2Int(1, 1);
    public Vector2Int stride;
    protected Vector2Int _oldDilution = new Vector2Int(1, 1);
    public Vector2Int dilution;

    protected bool _oldPadding = true;
    public bool padding;

    [Range(0.0f, 5.0f)]
    public float pointBrightness = 1.0f;

    [Range(0.0f, 5.0f)]
    public float weightBrightness = 1.0f;

    [Range(0.0f, 5.0f)]
    public float zOffset = 1.0f;

    [Range(-5.0f, 5.0f)]
    public float yOffset = 0f;

    protected Layer _inputLayer;
    protected Mesh _mesh;
    protected Renderer _renderer;

    private bool _initializedMesh = false;
    protected bool _initialized = false;

    private List<Layer> _observers = new List<Layer>();

    protected bool _renderColored = false;

    public int epoch = 0;

    public bool showOriginalDepth = false;

    protected List<FeatureMap> _featureMaps;
    protected abstract List<GridShape> GetPointShapes();
    public abstract Vector3Int GetOutputShape();
    protected abstract Vector3[] GetInterpolatedFeatureMapPositions();


    /// <summary>
    /// Initialize stuff. Replacement for constructor, checks if already initialized.
    /// </summary>
    public virtual void InitIfUnitialized()
    {
        if (_initialized)
            return;

        _renderer = GetComponent<Renderer>();

        if (!HasInitializedMesh())
            InitMesh();

        InitFeatureMapsForInputParams();

        _initialized = true;
    }

/// <summary>
/// Initialize Mesh, client should check
/// </summary>
    public virtual void InitMesh()
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();
        mf.sharedMesh = mesh;
        _mesh = mesh;
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        _initializedMesh = true;
    }

    /// <summary>
    /// Initializes feature map List according to this Layers input params (depends only on this layers properties).
    /// </summary>
    protected virtual void InitFeatureMapsForInputParams()
    {
        Vector3[] filterPositions = GetInterpolatedFeatureMapPositions();

        _featureMaps = new List<FeatureMap>();
        for (int i = 0; i < depth; i++)
        {
            FeatureMap map = new FeatureMap(this, i);
            _featureMaps.Add(map);
        }

    }

    /// <summary>
    /// Updates feature map list according to new Layer input parameters  (depends only on this layers properties).
    /// </summary>
    protected virtual void UpdateFeatureMapsForInputParams(bool topoChanged)
    {
        Vector3[] filterPositions = GetInterpolatedFeatureMapPositions();

        for (int i = 0; i < _featureMaps.Count; i++)
        {
            _featureMaps[i].UpdateValuesForInputParams(this, topoChanged);
        }
    }


    /// <summary>
    /// call after adding output layer or requesting output points of this layer with changed topology
    /// </summary>
    public virtual void InitFeatureMapsForOutputParams(InputAcceptingLayer outputLayer)
    {
        foreach (FeatureMap featureMap in _featureMaps)
        {
            featureMap.AddOutputLayer(outputLayer);
        }
    }

    /// <summary>
    /// call when requesting output points of this layer without changed topology
    /// </summary>
    public virtual void UpdateFeatureMapsForOutputParams(InputAcceptingLayer outputLayer, bool topoChanged)
    {
        foreach (FeatureMap featureMap in _featureMaps)
        {
            featureMap.UpdateForOutputLayer(outputLayer, topoChanged);
        }
    }

    private void RemoveOutputLayerFromFeaturemaps(InputAcceptingLayer outputLayer)
    {
        foreach (FeatureMap featureMap in _featureMaps)
        {
            featureMap.RemoveOutputLayer(outputLayer);
        }
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
    public virtual List<List<GridShape>> GetLineStartShapes(InputAcceptingLayer outputLayer, float allCalcs, int convLocation)
    {
        UpdateFeatureMapsForInputParams(true);

        List<List<GridShape>> filterGrids = new List<List<GridShape>>();
        for (int i = 0; i < _featureMaps.Count; i++)
        {
            filterGrids.Add(_featureMaps[i].GetFilterGrids(outputLayer, allCalcs, convLocation));
        }
        return filterGrids;
    }

    protected void Start()
    {
        InitIfUnitialized();

        CheckAndHandleParamChanges();
    }

    /// <summary>
    /// Called after each parameter change in the editor. Contains main update procedures.
    /// </summary>
    protected void OnValidate()
    {
        InitIfUnitialized();

        CheckAndHandleParamChanges();
    }

    public void OnInputParamChange(bool topoChanged)
    {
        if (topoChanged)
        {
            UpdateMeshTopoChanged();
        }
        else
        {
            UpdateMeshTopoUnchanged();
        }
    }

    protected virtual void CheckAndHandleParamChanges()
    {
        if (CheckTopologyChanged_OneValidCall())
        {
            UpdateMeshTopoChanged();
        } else
        {
            UpdateMeshTopoUnchanged();
        }
    }
   

    /// <summary>
    /// Gets called once per frame
    /// </summary>
    // Update is called once per frame
    void Update()
    {
        //UpdateMesh();
    }

    public virtual void UpdateMeshTopoChanged()
    {
        UpdateForChangedParams(true);

        RebuildMeshAndNotifyObservers(true);
    }

    public virtual void UpdateMeshTopoUnchanged()
    {
        UpdateForChangedParams(false);

        RebuildMeshAndNotifyObservers(false);
    }

    private void RebuildMeshAndNotifyObservers(bool topoChanged)
    {
        _mesh.Clear();
        CalcMesh();

        NotifyObservers(topoChanged);
    }

    protected virtual void UpdateForChangedParams(bool topoChanged)
    {
        UpdateFeatureMapsForInputParams(topoChanged);

    }

    /// <summary>
    /// Recursively returns the right ZOffset based on input layers.
    /// </summary>
    /// <returns>Z Base Position</returns>
    protected virtual float ZPosition()
    {
        if (_inputLayer != null) return _inputLayer.ZPosition() + zOffset;
        else return 0;
    }

    protected virtual float YPosition()
    {
        if (_inputLayer != null) return _inputLayer.YPosition() + yOffset;
        else return 0;
    }

    /// <summary>
    /// Just returns a Vector on the Z Axis with the correct z positon.
    /// </summary>
    /// <returns></returns>
    public Vector3 CenterPosition()
    {
        return new Vector3(0, YPosition(), ZPosition());
    }

    /// <summary>
    /// Calculates rounded padding based on filter size.
    /// </summary>
    /// <returns></returns>
    public Vector2Int GetPadding()
    {
        return padding ? Vector2Int.FloorToInt(convShape / new Vector2(2, 2)) : new Vector2Int(0, 0);
    }

    /// <summary>
    /// Returns a 2d Vector of the layers output shape.
    /// </summary>
    /// <returns></returns>
    public virtual Vector2Int Get2DOutputShape()
    {
        return new Vector2Int(1, 1);
    }

    /// <summary>
    /// Calculates and sets the mesh of the Layers game object.
    /// </summary>
    public virtual void CalcMesh()
    {
        _mesh.subMeshCount = 2;

        List<Vector3> verts = new List<Vector3>();
        List<int> inds = new List<int>();

        AddNodes(verts, inds);


        _mesh.SetVertices(verts);
        _mesh.SetIndices(inds.ToArray(), MeshTopology.Points, 0);

    }

    /// <summary>
    /// Adds the "pixels" of the feature maps to the mesh.
    /// </summary>
    /// <param name="verts"></param>
    /// <param name="inds"></param>
    protected virtual void AddNodes(List<Vector3> verts, List<int> inds)
    {
        Vector3 zPos = CenterPosition();

        foreach (Shape shape in GetPointShapes())
        {
            Vector3[] v = shape.GetVertices(true);

            for (int i = 0; i < v.Length; i++)
            {
                verts.Add(v[i] + zPos);
                inds.Add(inds.Count); 
            }
        }

    }

    public virtual void AddOuputLayer(InputAcceptingLayer outputLayer)
    {
        InitFeatureMapsForOutputParams(outputLayer);
    }

    public virtual void RemoveOutputLayer(InputAcceptingLayer outputLayer)
    {
        RemoveOutputLayerFromFeaturemaps(outputLayer);
    }

    /// <summary>
    /// Adds an observer to be notified on parameter changes on this layer to allow downstream updated of dependent layers.
    /// </summary>
    /// <param name="observer"></param>
    public void AddObserver(Layer observer)
    {
        if(! _observers.Contains(observer))
        {
            _observers.Add(observer);
        }
    }

    /// <summary>
    /// Removes an observer.
    /// </summary>
    /// <param name="observer"></param>
    public void RemoveObserver(Layer observer)
    {
        if (_observers.Contains(observer))
        {
            _observers.Remove(observer);
        }
    }

    /// <summary>
    /// Notifies obsevers that a parameter change has occured.
    /// </summary>
    public void NotifyObservers(bool topoChanged)
    {
        foreach(Layer observer in _observers)
        {
            observer.OnInputParamChange(topoChanged);
        }
    }

    /// <summary>
    /// Get the input layer of this Layer.
    /// </summary>
    /// <returns></returns>
    public Layer GetInputLayer()
    {
        return _inputLayer;
    }

    /// <summary>
    /// Set the epoch of this layer to load the correct tensor values in the mesh calculation procedure.
    /// </summary>
    /// <param name="epoch"></param>
    public void SetEpoch(int epoch)
    {
        this.epoch = epoch;
        OnValidate();
    }

    public bool HasInitializedMesh()
    {
        return _initializedMesh;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected virtual bool CheckTopologyChanged_OneValidCall()
    {
        bool topologyChanged = false;

        topologyChanged |= CheckConvShapeChanged_OneValidCall();
        topologyChanged |= CheckStrideChanged_OneValidCall();
        topologyChanged |= CheckDilutionChanged_OneValidCall();
        topologyChanged |= CheckPaddingChanged_OneValidCall();
        topologyChanged |= CheckDepthChanged_OneValidCall();

        return topologyChanged;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckConvShapeChanged_OneValidCall()
    {
        if (convShape != _oldConvShape)
        {
            _oldConvShape = convShape;
            return true;
        }
        return false;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckStrideChanged_OneValidCall()
    {
        if (stride != _oldStride)
        {
            _oldStride = stride;
            return true;
        }
        return false;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckDilutionChanged_OneValidCall()
    {
        if (dilution != _oldDilution)
        {
            _oldDilution = dilution;
            return true;
        }
        return false;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckPaddingChanged_OneValidCall()
    {
        if (padding != _oldPadding)
        {
            _oldPadding = padding;
            return true;
        }
        return false;
    }

    /// <summary>
    /// checks param and sets old to new, only call once for each change!
    /// </summary>
    /// <returns></returns>
    protected bool CheckDepthChanged_OneValidCall()
    {
        if (depth != _oldDepth)
        {
            _oldDepth = depth;
            return true;
        }
        return false;
    }


    public virtual FeatureMapInputProperties GetFeatureMapInputProperties(int featureMapIndex)
    {
        return new FeatureMapInputProperties();
    }
}
