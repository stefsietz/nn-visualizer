Shader "Unlit/ColoredLine"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_width ("width", Range(0, 0.2)) = 0.05
		_npts("n points", Range(2, 20)) = 10
		_pow("curve power", Range(0.001, 5.0)) = 2.0
		_line_gradient_pow("line gradient power", Range(0.2, 10.0)) = 5.0
		_dark("dark color", Range(0.0, 1.0)) = 0.1
		_bright("bright color", Range(0.0, 1.0)) = 0.5
		_persp_width("perspective width", Range(0.0, 1.0)) = 1.0
	}
	SubShader
	{
		Tags {  "Queue"="Transparent" "RenderType"="Opaque" }
		LOD 100

		//ZWrite Off
		//ZTest Always
		//Blend SrcAlpha OneMinusSrcAlpha


		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma geometry geo
			#pragma fragment frag
			// make fog work
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float4 color : COLOR;
				float2 uv : TEXCOORD0;
				uint id : SV_VertexID;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float2 scrPos : TEXCOORD1;
				float4 vertex : SV_POSITION;
			};

			struct fragOut
			{
				fixed4 color : SV_Target;
				float depth: SV_Depth;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			float _width;
			int _npts;
			float _pow;
			float _persp_width;
			float _line_gradient_pow;
			float _dark;
			float _bright;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = v.vertex;
				o.uv.x = v.color.x;
				return o;
			}

			v2f VertexOutput(float4 wpos, float2 uv, v2f input)
			{
				v2f o = input;
				o.vertex = wpos;
				o.uv = uv;
				o.scrPos = ComputeScreenPos(wpos);
				return o;
			}

			float log(float in_value){
				return 1.0 / (1 + exp(-_pow*4*(in_value-0.5)));
			}

			float4 p_lookup(float4 p0, float4 p1, float t) {
				float t_out = log(t);
				float min = log(0.0);
				float max = log(1.0);
				t_out = (t_out - min) / (max - min);

				float4 diff = p1 - p0;

				float4 o = float4(p0.xy + diff.xy * t_out, p0.z + diff.z * t, p0.w + diff.w * t);
				return o;
			}

			[maxvertexcount(120)]
			void geo(
				line v2f input[2],
				inout TriangleStream<v2f> outStream
			)
			{
				// Vertex inputs
				float4 wp0 = input[0].vertex;
				float4 wp1 = input[1].vertex;


				for(int i=0; i<_npts-1; i++) {
					float4 c0 = p_lookup(wp0, wp1, i/(float)(_npts-1));
					float4 c1 = p_lookup(wp0, wp1, (i+1)/(float)(_npts-1));

					c0 = UnityObjectToClipPos(c0);
					c1 = UnityObjectToClipPos(c1);
					if(c0.w < -0.0 || c1.w <-0.0)
						continue;

					float2 lineVec = normalize(c1.xy - c0.xy);
					float2 normalVec = lineVec.yx * float2(1.0,-1.0);
					float2 stretch = float2(_ScreenParams.y / _ScreenParams.x, 1.0);
					normalVec = normalize(normalVec)*_width*0.5*stretch / ((1.0-_persp_width) + _persp_width *(c0.w+c1.w)/2.0);

					c0 = c0/c0.w;
					c1 = c1/c1.w;

					outStream.Append(VertexOutput(c0 - float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 0.0), input[0]));
					outStream.Append(VertexOutput(c0 + float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 1.0), input[0]));
					outStream.Append(VertexOutput(c1 - float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 0.0), input[1]));
					outStream.RestartStrip();
					outStream.Append(VertexOutput(c1 - float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 0.0), input[1]));
					outStream.Append(VertexOutput(c0 + float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 1.0), input[0]));
					outStream.Append(VertexOutput(c1 + float4(normalVec, 0.0, 0.0), float2(input[1].uv.x, 1.0), input[1]));
					outStream.RestartStrip();
				}
				
			}

			float3 colLookup(float value) {
				float3 red = float3(1.0, 0, 0);
				float3 blue = float3(0, 0, 1.0);
				float3 white = float3(1.0, 1.0, 1.0);

				if (value > 0.0) {
					return lerp(white, red, value);
				}
				else {
					return lerp(white, blue, -value);
				}
			}

			float4 color(float2 uv) {
				float cdist = abs(uv.y-0.5);
				float bright = pow(cdist * 2.0, 2);//(cdist < 0.25 ? 0.0 : 1.0);
				bright = clamp(1.0-bright, 0.0, 1.0);
				bright = pow(bright, _line_gradient_pow);
				bright = lerp(_dark, _bright, bright);
				float3 col = colLookup(uv.x) * bright;
				float4 c = float4(col, 1.0);
				return c;
			}
			
			fragOut frag (v2f i)
			{
				fragOut o;
				// sample the texture
				fixed4 col = color(i.uv);

				o.color = col;
				o.depth = i.vertex.z/i.vertex.w-0.0001;

				return o;
			}

			ENDCG
		}

	}
}
