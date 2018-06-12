using UnityEngine;

public class GridShape : Shape
{
    public Vector2Int resolution;
    public Vector2 spacing;

    public GridShape(Vector3 position, Vector2Int resolution, Vector2 spacing) : base(position)
    {
        this.resolution = resolution;
        this.spacing = spacing;

        initVerts();
    }

    public GridShape InterpolatedGrid(GridShape target, float alpha)
    {
        if(this.resolution != target.resolution)
        {
            throw new System.Exception("Grid Resolution has to match!");
        }
        Vector3 pos = this.position * (1.0f - alpha) + target.position * alpha;
        Vector2 spacing = this.spacing * (1.0f - alpha) + target.spacing * alpha;

        return new GridShape(pos, this.resolution, spacing);
    }

    private void initVerts()
    {
        _verts = new Vector3[resolution.x * resolution.y];
        for (int i=0; i<_verts.Length; i++)
        {
            _verts[i] = new Vector3(0, 0, 0);
        }
        calcVertices();
    }

    protected override void calcVertices()
    {
        for (int i = 0; i < resolution.x; i++)
        {
            for (int j = 0; j < resolution.y; j++)
            {
                Vector2 offset = new Vector2((resolution.x - 1) * spacing.x / 2, (resolution.y - 1) * -spacing.y / 2);
                float x = position.x + i * spacing.x - offset.x;
                float y = position.y + j * -spacing.y - offset.y;

                _verts[i * resolution.y + j].Set(x, y, position.z);
            }
        }
    }

    public override float[] Bbox()
    {
        Vector2 offset = new Vector2((resolution.x - 1) * spacing.x / 2, (resolution.y - 1) * spacing.y / 2);

        float[] o = { position.x - offset.x, position.y - offset.y, position.x + offset.x, position.y + offset.y};

        return o;
    }

    public Vector3[] BboxV(float zpos)
    {
        float[] bbox = Bbox();
        Vector3 p1 = new Vector3(bbox[0], bbox[1], position.z + zpos);
        Vector3 p2 = new Vector3(bbox[2], bbox[1], position.z + zpos);
        Vector3 p3 = new Vector3(bbox[2], bbox[3], position.z + zpos);
        Vector3 p4 = new Vector3(bbox[0], bbox[3], position.z + zpos);

        Vector3[] o = new Vector3[4];
        o[0] = p1;
        o[1] = p2;
        o[2] = p3;
        o[3] = p4;

        return o;
    }

    public static Vector3[] ScaledUnitGrid(int xres, int yres, Vector3 pos, float scale)
    {
        Vector3[] o = new Vector3[xres * yres];
        float xoffset = (xres-1.0f) / 2.0f;
        float yoffset = (yres - 1.0f) / 2.0f;

        for (int i = 0; i < xres; i++)
        {
            for (int j = 0; j < yres; j++)
            {
                o[i * yres + j] = new Vector3((i - xoffset + pos.x) * scale, (j - yoffset + pos.y) * scale, pos.z);
            }
        }
        return o;
    }
}