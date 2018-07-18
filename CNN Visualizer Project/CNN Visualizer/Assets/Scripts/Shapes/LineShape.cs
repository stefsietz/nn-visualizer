using UnityEngine;

public class LineShape : Shape
{
    public int resolution;
    public float spacing;

    public LineShape(Vector3 position, int resolution, float spacing) : base(position)
    {
        this.resolution = resolution;
        this.spacing = spacing;

        InitVerts();
    }

    public override object Clone()
    {
        return new LineShape(this.position, this.resolution, this.spacing);
    }


    protected override void InitVerts()
    {
        _verts = new Vector3[resolution];
        for (int i=0; i<_verts.Length; i++)
        {
            _verts[i] = new Vector3(0, 0, 0);
        }
        CalcVertices();
    }

    protected override void CalcVertices()
    {
        for (int i = 0; i < resolution; i++)
        {
            float offset = (resolution - 1) * spacing / 2.0f;
            float x = position.x + i * spacing - offset;

            _verts[i].Set(x, position.y, position.z);
        }
    }

    public override float[] GetBbox()
    {
        float offset = (resolution - 1) * spacing / 2.0f;

        float[] o = {position.x-offset, position.y, position.x + offset, position.y};

        return o;
    }

    public static Vector3[] ScaledUnitLine(int res, Vector3 pos, Vector3 dir, float scale)
    {
        Vector3[] o = new Vector3[res];
        float offset = (res-1.0f) / 2.0f;

        dir.Normalize();
        dir *= scale;

        for (int i = 0; i < res; i++)
        {
             o[i] = pos + (i - offset + pos.x) * dir;
        }
        return o;
    }
}