using UnityEngine;

public class CircleShape : Shape
{
    public int resolution;
    public float spacing;

    public CircleShape(Vector3 position, int resolution, float spacing) : base(position)
    {
        this.resolution = resolution;
        this.spacing = spacing;

        InitVerts();
    }

    public override object Clone()
    {
        return new CircleShape(this.position, this.resolution, this.spacing);
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
            float x = position.x + Mathf.Cos(i/ (float)resolution *Mathf.PI*2) * spacing;
            float y = position.y + Mathf.Sin(i / (float)resolution * Mathf.PI * 2) * spacing;

            _verts[i].Set(x, y, position.z);
        }
    }

    public override float[] GetBbox()
    {
        float[] o = { position.x - spacing, position.y - spacing, position.x + spacing, position.y + spacing };

        return o;
    }

    public static Vector3[] ScaledUnitCircle(int res, Vector3 pos, float scale)
    {
        Vector3[] o = new Vector3[res];

        for (int i = 0; i < res; i++)
        {
            float x = pos.x + Mathf.Cos(i /(float) res * Mathf.PI * 2) * scale;
            float y = pos.x + Mathf.Sin(i / (float)res * Mathf.PI * 2) * scale;
            o[i] = new Vector3(x, y, pos.z);
        }
        return o;
    }
}