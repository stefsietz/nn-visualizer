Shader "Unlit/ColoredSphere"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_width ("width", Range(0, 0.2)) = 0.05
		[Toggle] _useBlueRedCmap("Use Blue/Red CMAP", Int) = 1
		[Toggle] _squarePixels("Square Pixels", Int) = 0
	}
	SubShader
	{
		Tags { "Queue"="Transparent" "RenderType"="Transparent" }
		LOD 100

		//ZWrite Off
		Blend SrcAlpha OneMinusSrcAlpha

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma geometry geo
			#pragma fragment frag
			#pragma require geometry
			
			#include "UnityCG.cginc"
			 
			struct appdata
			{
				float4 vertex : POSITION;
				float4 color : COLOR;
				float2 uv : TEXCOORD0;
				uint   id  : SV_VertexID;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float4 col : COLOR;
			};

			sampler2D _MainTex;
			float _width;
			bool _useBlueRedCmap;
			bool _squarePixels;
			
			v2f vert (appdata v, uint vid : SV_VertexID)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				float2 uv = float2(0.5, 0.5);
				o.col = v.color;

				return o;
			}

			v2f VertexOutput(float4 wpos, float2 uv, v2f input)
			{
				v2f o = input;
				o.vertex = wpos;
				o.uv = uv;
				o.col = input.col;
				return o;
			}

			[maxvertexcount(6)]
			void geo(
				point v2f input[1],
				inout TriangleStream<v2f> outStream
			)
			{
				// Vertex inputs
				float4 wp0 = input[0].vertex;
				float2 stretch = float2(_ScreenParams.y / _ScreenParams.x, 1.0);
				float2 lt = float2(-0.5, 0.5)*_width* stretch;
				float2 lb = float2(-0.5, -0.5)*_width* stretch;
				float2 rb = float2(0.5, -0.5)*_width* stretch;
				float2 rt = float2(0.5, 0.5)*_width* stretch;

				float2 ltuv = float2(0.0, 1.0);
				float2 lbuv = float2(0.0, 0.0);
				float2 rbuv = float2(1.0, 0.0);
				float2 rtuv = float2(1.0, 1.0);

				outStream.Append(VertexOutput(wp0 + float4(lt, 0.0, 0.0), ltuv, input[0]));
				outStream.Append(VertexOutput(wp0 + float4(lb, 0.0, 0.0), lbuv, input[0]));
				outStream.Append(VertexOutput(wp0 + float4(rt, 0.0, 0.0), rtuv, input[0]));
				outStream.RestartStrip();
				outStream.Append(VertexOutput(wp0 + float4(rt, 0.0, 0.0), rtuv, input[0]));
				outStream.Append(VertexOutput(wp0 + float4(lb, 0.0, 0.0), lbuv, input[0]));
				outStream.Append(VertexOutput(wp0 + float4(rb, 0.0, 0.0), rbuv, input[0]));
				outStream.RestartStrip();
			}

			float brightness(float2 uv) {
				if (_squarePixels) {
					return 1.0;
				}
				else {
					float o = clamp(1.0 - distance(float2(0.5, 0.5), uv)*2.0, 0.0, 1.0);
					o = pow(o, 0.7);
					return o;
				}
			}

			float alpha(float brightness) {
			
				float o = pow(brightness, 0.1);
				return o;
			}

			float3 colLookup(float3 col) {
				float3 red = float3(1.0, 0, 0);
				float3 blue = float3(0, 0, 1.0);
				float3 white = float3(1.0, 1.0, 1.0);

				if (!_useBlueRedCmap) {
					return col;
				}
				else if (col.x > 0.0) {
					return lerp(white, red, col.x);
				}
				else {
					return lerp(white, blue, -col.x);
				}
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float bright = brightness(i.uv);

				float3 color = colLookup(i.col.rgb) * bright;

				fixed4 col = float4(color.r*bright, color.g*bright, color.b*bright, alpha(bright));

				return col;
			}
			ENDCG
		}
	}
}
