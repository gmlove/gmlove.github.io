precision highp float;

varying vec2 vTextureCoord;
uniform sampler2D uSampler;

void main(void)
{
   vec2 uvs = vTextureCoord.xy;
   vec4 fg = texture2D(uSampler, vTextureCoord);

   float t = 0.12;
   if (fg.r < t && fg.g < t && fg.b < t) {
        fg.a = 0.0;
   }

   gl_FragColor = fg;
}