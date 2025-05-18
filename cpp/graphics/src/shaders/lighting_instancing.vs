#version 330

/* ---- fixed attributes that every raylib mesh provides ------------- */
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;   // must be *used* to stay alive

/* ---- per-instance matrix occupies 4-7 ----------------------------- */
layout(location = 4) in mat4 instanceTransform;

uniform mat4 mvp;

out vec3 FragPos;
out vec3 FragNrm;
out vec4 FragCol;          // keep colour alive → stops optimiser dropping it

void main()
{
    // If DrawMesh() is used the attribute is unbound; detect that.
    mat4 model = (instanceTransform[3][3] == 0.0) ? mat4(1.0)
                                                 : instanceTransform;

    vec4 world = model * vec4(vertexPosition, 1.0);
    FragPos = world.xyz;
    FragNrm = mat3(transpose(inverse(model))) * vertexNormal;
    FragCol = vertexColor;         // ← USE the colour once, nothing more

    gl_Position = mvp * world;
}