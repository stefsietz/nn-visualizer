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
    public Vector2Int reducedShape;
    public Vector2Int stride = new Vector2Int(1, 1);
    public bool padding = true;

    [Range(0.0f, 5.0f)]
    public float pointBrightness = 1.0f;

    [Range(0.0f, 5.0f)]
    public float zOffset = 1.0f;

    protected Layer _inputLayer;
    protected Mesh _mesh;
    protected Renderer _renderer;

    private bool _initialized = false;

    private List<Layer> observers = new List<Layer>();

    protected bool renderColored = false;

    public int epoch = 0;

    public bool showOriginalDepth = false;

    protected abstract List<Shape> GetPointShapes();
    public abstract List<List<Shape>> GetLineStartShapes(Vector2Int convShape, Vector2Int outputShape, Vector2 theoreticalOutputShape, Vector2Int stride, float allCalcs);
    public abstract Vector3Int GetOutputShape();

    public Layer()
    {

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
        UpdateForChangedParams();
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
        return Vector2Int.FloorToInt(reducedShape / new Vector2(2, 2));
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
        if(! observers.Contains(observer))
        {
            observers.Add(observer);
        }
    }

    /// <summary>
    /// Removes an observer.
    /// </summary>
    /// <param name="observer"></param>
    public void RemoveObserver(Layer observer)
    {
        if (observers.Contains(observer))
        {
            observers.Remove(observer);
        }
    }

    /// <summary>
    /// Notifies obsevers that a parameter change has occured.
    /// </summary>
    public void NotifyObservers()
    {
        foreach(Layer observer in observers)
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
}
