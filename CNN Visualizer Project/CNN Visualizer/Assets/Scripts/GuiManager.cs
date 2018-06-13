using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

/*! \mainpage Index Page
 * 
 * \section Introduction
 * \dir src/
 */

/// <summary>
/// Acts as an interface between the GUI and other classes.
/// </summary>
public class GuiManager : MonoBehaviour
{

    public float convFullresHeight = 5.0f;

    public Text predictedLabel;

    public Material lineMaterial;

    public Material pixelMaterial;

    private int epoch = 0;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnValidate()
    {
        GlobalManager.Instance.SetFullResHeight(convFullresHeight);
    }

    /// <summary>
    /// Logs Layer names for Debugging
    /// </summary>
    public void LogLayers()
    {
        foreach(Layer l in GlobalManager.Instance.GetAllLayersOrdered())
        {
            Debug.Log(l.name);
        }
    }

    /// <summary>
    /// Sets the epoch to be loaded.
    /// </summary>
    /// <param name="value"></param>
    public void SetEpoch(float value)
    {
        epoch = (int)value;
        GlobalManager.Instance.SetEpoch(epoch);
        predictedLabel.text = GlobalManager.Instance.predPerEpoch[epoch];
    }

    /// <summary>
    /// Set the line width of the edges.
    /// </summary>
    /// <param name="value"></param>
    public void SetLineWidth(float value)
    {
        lineMaterial.SetFloat("_width", value);
    }

    /// <summary>
    /// Sets the brightness of all edges (color contrast when rendering red/blue mapping)
    /// </summary>
    /// <param name="value"></param>
    public void SetWeightBrightness(float value)
    {
        GlobalManager.Instance.SetWeightBrightness(value);
    }

    /// <summary>
    /// Sets the brightness of all nodes (color contrast when rendering red/blue mapping)
    /// </summary>
    /// <param name="value"></param>
    public void SetPointBrightness(float value)
    {
        GlobalManager.Instance.SetPointBrightness(value);
    }

    public void SetExpansionLevel(float value)
    {
        GlobalManager.Instance.SetExpansionLevel(value);
    }

    public void SetFullres(bool value)
    {
        GlobalManager.Instance.SetFullResDisplay(value);
    }

    public void SetSquarePixels(bool value)
    {
        pixelMaterial.SetInt("_squarePixels", value ? 1 : 0);
    }

    public void SetBWPixels(bool value)
    {
        pixelMaterial.SetInt("_useBlueRedCmap", value ? 0 : 1);
    }

    /// <summary>
    /// Load the currently selected Testsample epochs
    /// </summary>
    public void Load()
    {
        TFLoader loader = GameObject.FindObjectOfType<TFLoader>();
        loader.Load();
    }

}
