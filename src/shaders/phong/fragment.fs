#version 330 core

struct PointLight {
    vec3 position;
    vec3 color;
    float strength;
};

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

uniform sampler2D imageTexture;
uniform PointLight Lights[8];
uniform vec3 cameraPosition;

uniform float ks;
uniform float kd;
uniform float ka;
uniform float alfa;

out vec4 color;

vec3 calculatePointLight(PointLight light, vec3 fragPosition, vec3 fragNormal);

void main()
{
    //ambient
    vec3 temp = ka * texture(imageTexture, fragmentTexCoord).rgb;

    for (int i = 0; i < 8; i++) {
        temp += calculatePointLight(Lights[i], fragmentPosition, fragmentNormal);
    }

    color = vec4(temp, 1);
}

vec3 calculatePointLight(PointLight light, vec3 fragPosition, vec3 fragNormal) {

    vec3 baseTexture = texture(imageTexture, fragmentTexCoord).rgb;
    vec3 result = vec3(0);

    //geometric data
    vec3 fragLight = light.position - fragPosition;
    float distance = length(fragLight);
    fragLight = normalize(fragLight);
    vec3 fragCamera = normalize(cameraPosition - fragPosition);
    vec3 halfVec = normalize(fragLight + fragCamera);

    //diffuse
    result += light.color * light.strength * (kd* max(0.0, dot(fragNormal, fragLight))) / (distance * distance) * baseTexture;

    //specular
    result += light.color * light.strength * (ks *(pow(max(0.0, dot(fragNormal, halfVec)),alfa))) / (distance * distance);

    return result;
}