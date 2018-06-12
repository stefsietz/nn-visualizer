﻿using UnityEngine;
using System;
using System.Collections.Generic;


/// <summary>
/// Base class for various Shape Subclasses that are used for calculating vertex positions based on a center position and size.
/// </summary>
public abstract class Shape
{
    public Vector3 position { get; set; }
    protected Vector3[] _verts;

    public Shape(Vector3 position)
	{
        this.position = position;
	}

    /// <summary>
    /// Return vertices of this shape. If update parameter is false, stored vertices are returned, else they are recalculated.
    /// </summary>
    /// <param name="update"></param>
    /// <returns></returns>
    public Vector3[] GetVertices(bool update)
    {
        if (update) calcVertices();

        return _verts;
    }

    /// <summary>
    /// returns 3d Vectors that can be added to the mesh.
    /// </summary>
    protected abstract void calcVertices();

    /// <summary>
    /// calculate the bounding box of this shape
    /// </summary>
    /// <returns></returns>
    public abstract float[] Bbox();

    /// <summary>
    /// Interpolate 2 Shapes with different parameters and return their vertices.
    /// </summary>
    /// <param name="s1"></param>
    /// <param name="s2"></param>
    /// <param name="alpha"></param>
    /// <returns></returns>
    public static Vector3[] InterpolateShapes(Vector3[] s1, Vector3[] s2, float alpha)
    {
        Vector3[] o = new Vector3[s1.Length];
        for(int i=0; i<s1.Length; i++)
        {
            o[i] = s1[i] * (1.0f - alpha) + s2[i] * alpha;
        }
        return o;
    }
}