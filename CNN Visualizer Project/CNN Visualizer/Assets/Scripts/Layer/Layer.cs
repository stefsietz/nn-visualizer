using UnityEngine;
using System.Collections;
using System.Collections.Generic;

// Runtime code here
#if UNITY_EDITOR
[ExecuteInEditMode]
#endif
// Runtime code here
[RequireComponent(typeof(MeshFilter))]

/// Base class for the layer component
public abstract class Layer : MonoBehaviour
{
    public delegate void OnTopologyChangeDelegate();
    public event OnTopologyChangeDelegate OnTopologyChange;

    //TODO: these should be more class specific
    protected Vector2Int _oldConvShape;
    public Vector2Int convShape
    {
        get
        {
            return _oldConvShape;
        }
        set
        {
            if (_oldConvShape == value)
                return;

            _oldConvShape = value;

            RaiseOnTopologyChange();
        }
    }

    protected Vector2Int _oldStride = new Vector2Int(1, 1);
    public Vector2Int stride
    {
        get
        {
            return _oldStride;
        }
        set
        {
            if (_oldStride == value)
                return;

            _oldStride = value;

            RaiseOnTopologyChange();
        }
    }
    protected Vector2Int _oldDilution = new Vector2Int(1, 1);
    public Vector2Int dilution
    {
        get
        {
            return _oldDilution;
        }
        set
        {
            if (_oldDilution == value)
                return;

            _oldDilution = value;

            RaiseOnTopologyChange();
        }
    }

    protected bool _oldPadding = true;
    public bool padding
    {
        get
        {
            return _oldPadding;
        }
        set
        {
            if (_oldPadding == value)
                return;

            _oldPadding = value;

            RaiseOnTopologyChange();
        }
    }

    [Range(0.0f, 5.0f)]
    public float pointBrightness = 1.0f;

    [Range(0.0f, 5.0f)]
    public float weightBrightness = 1.0f;

    [Range(0.0f, 5.0f)]
    public float zOffset = 1.0f;

    protected Layer _inputLayer;
    protected Mesh _mesh;
    protected Renderer _renderer;

    private bool _initialized = false;

    private List<Layer> _observers = new List<Layer>();

    protected bool _renderColored = false;

    public int epoch = 0;

    public bool showOriginalDepth = false;

    protected abstract List<Shape> GetPointShapes();
    public abstract List<List<Shape>> GetLineStartShapes(InputAcceptingLayer ouputLayer, float allCalcs);
    public abstract Vector3Int GetOutputShape();

    public Layer()
    {
        OnTopologyChange += new OnTopologyChangeDelegate(UpdateForChangedParams);
    }

    protected void RaiseOnTopologyChange()
    {
        if (OnTopologyChange != null)
            OnTopologyChange();
    }

    public virtual void Init()
    {
        if (_initialized)
            return;

        MeshFilter mf = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();
        mf.sharedMesh = mesh;
        _mesh = mesh;
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        _initialized = true;

        UpdateMesh();
    }

    protected void Start()
    {
        Init();
    }

    /// <summary>
    /// Called after each parameter change in the editor. Contains main update procedures.
    /// </summary>
    protected void OnValidate()
    {
        _renderer = GetComponent<Renderer>();
        Init();
        UpdateMesh();
    }

    public virtual void UpdateMesh()
    {
        if (_mesh != null)
        {
            _mesh.Clear();
            CalcMesh();
        }
        NotifyObservers();
    }

    protected virtual void UpdateForChangedParams()
    {
    }

    /// <summary>
    /// Recursively returns the right ZOffset based on input layers.
    /// </summary>
    /// <returns>Z Base Position</returns>
    protected float ZPosition()
    {
        if (_inputLayer != null) return _inputLayer.ZPosition() + zOffset;
        else return 0;
    }

    /// <summary>
    /// Just returns a Vector on the Z Axis with the correct z positon.
    /// </summary>
    /// <returns></returns>
    public Vector3 CenterPosition()
    {
        return new Vector3(0, 0, ZPosition());
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
        Vector3 zPos = new Vector3(0, 0, ZPosition());

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

    /// <summary>
    /// Gets called once per frame
    /// </summary>
    // Update is called once per frame
    void Update()
    {
        if (!_initialized)
        {
            Init();
        }
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
    public void NotifyObservers()
    {
        foreach(Layer observer in _observers)
        {
            if (observer != null)
            {
                if(observer._mesh == null)
                {
                    observer.Init();
                }
                observer.OnValidate();
            }
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

    public bool IsInitialized()
    {
        return _initialized;
    }

    public virtual List<List<Shape>> GetLineStartShapes(InputAcceptingLayer outputLayer, float allCalcs, int convLocation)
    {
        return GetLineStartShapes(outputLayer, allCalcs);
    }
}
