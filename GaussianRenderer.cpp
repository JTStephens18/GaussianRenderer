// GaussianRenderer.cpp : Defines the entry point for the application.

#include "GaussianRenderer.h"
#include "./Sphere.h"
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h> 
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, auto& cloud);
//void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
//void cursor_position_callback(GLFWwindow* window, double xpos, double ypos); 

const unsigned int SCREEN_WIDTH = 800;
const unsigned int SCREEN_HEIGHT = 600;

const unsigned int numInstancesCount = 1000;

glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
//glm::vec3 cameraPos = glm::vec3(-321.6, -40.0f, 5355.0f);
//glm::vec3 cameraFront = glm::vec3(9.33f, -3.11f, -252.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.f;
float pitch = 0.0f;
float lastX = SCREEN_WIDTH / 2.0;
float lastY = SCREEN_HEIGHT / 2.0;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

std::vector<size_t> sortedIdx;
const float C0 = 0.28209479177387814f;


//pcl::PointCloud<pcl::PointXYZ> cloud;

// Define custom point type with exact property names
struct GaussianData {
	PCL_ADD_POINT4D;                  // Quad-word XYZ + padding

	float nx, ny, nz;                 // Normal vectors
	float f_dc_0, f_dc_1, f_dc_2;
	float opacity;
	float scale_0, scale_1, scale_2;
	float rot_0, rot_1, rot_2, rot_3;

	PCL_MAKE_ALIGNED_OPERATOR_NEW     // Ensures proper alignment with Eigen
};

// Register the custom point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(
	GaussianData,
	(float, x, x) (float, y, y) (float, z, z)
	(float, nx, nx) (float, ny, ny) (float, nz, nz)
	(float, f_dc_0, f_dc_0) (float, f_dc_1, f_dc_1) (float, f_dc_2, f_dc_2)
	(float, opacity, opacity)
	(float, scale_0, scale_0) (float, scale_1, scale_1) (float, scale_2, scale_2)
	(float, rot_0, rot_0) (float, rot_1, rot_1) (float, rot_2, rot_2) (float, rot_3, rot_3)
)

std::vector<int> sortedGaussianIndices(numInstancesCount);
std::vector<float> viewDepth(numInstancesCount);

std::vector<glm::vec4> normRots;
std::vector<glm::vec3> expScales;

std::vector<std::vector<float>> cov3D;
std::vector<glm::mat3> Vrks;

std::vector<glm::vec4> calculateRotationNorms(const std::vector<glm::vec4>& rots) {
	std::vector<glm::vec4> norms;
	norms.reserve(rots.size());

	for (const auto& vec : rots) {
		float norm = glm::length(vec) + 1e-9f;
		glm::vec4 normVec = vec / norm;
		norms.push_back(normVec);
	}
	return norms;
}

glm::vec4 normalizeRotation(glm::vec4& rot) {
	float sumOfSqaures = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w;
	float normalizedVal = std::sqrt(sumOfSqaures);
	return glm::vec4(rot.w / normalizedVal, rot.x / normalizedVal, rot.y / normalizedVal, rot.z / normalizedVal);
	/*
	glm::vec4(
		glm::clamp((rot.x / normalizedVal) * 128.0f + 128.0f, 0.0f, 255.0f),
		glm::clamp((rot.y / normalizedVal) * 128.0f + 128.0f, 0.0f, 255.0f),
		glm::clamp((rot.z / normalizedVal) * 128.0f + 128.0f, 0.0f, 255.0f),
		glm::clamp((rot.w / normalizedVal) * 128.0f + 128.0f, 0.0f, 255.0f)
	);
	*/

	/*
	glm::vec4(
		glm::clamp(((rot.x / normalizedVal) - 128.0f) / 128.0f, 0.0f, 255.0f),
		glm::clamp(((rot.y / normalizedVal) - 128.0f) / 128.0f, 0.0f, 255.0f),
		glm::clamp(((rot.z / normalizedVal) - 128.0f) / 128.0f, 0.0f, 255.0f),
		glm::clamp(((rot.w / normalizedVal) - 128.0f) / 128.0f, 0.0f, 255.0f)
	);
	*/
};

void logRotations(std::vector<glm::vec4>& newRots) {
	std::cout << "New rots size " << newRots.size() << std::endl;
	std::cout << "New rots " << newRots[0].x << ", " << newRots[0].y << ", " << newRots[0].z << ", " << newRots[0].w << std::endl;

	glm::vec4 minVec(std::numeric_limits<float>::max());
	glm::vec4 maxVec(std::numeric_limits<float>::lowest());

	// Loop through all the glm::vec4's in the vector
	for (const auto& vec : newRots) {
		minVec = glm::min(minVec, vec);
		maxVec = glm::max(maxVec, vec);
	}

	// Print out the results
	std::cout << "Min Value: (" << minVec.x << ", " << minVec.y << ", " << minVec.z << ", " << minVec.w << ")" << std::endl;
	std::cout << "Max Value: (" << maxVec.x << ", " << maxVec.y << ", " << maxVec.z << ", " << maxVec.w << ")" << std::endl;
};

std::vector<float> computeCov3D(glm::vec4& rots, glm::vec3& scales) {

	glm::vec3 firstRow = glm::vec3(
		1.f - 2.f * (rots.z * rots.z + rots.w * rots.w), // First element of row 0
		2.f * (rots.y * rots.z - rots.x * rots.w),       // Second element of row 0
		2.f * (rots.y * rots.w + rots.x * rots.z)        // Third element of row 0
	);

	glm::vec3 secondRow = glm::vec3(
		2.f * (rots.y * rots.z + rots.x * rots.w),       // First element of row 1
		1.f - 2.f * (rots.y * rots.y + rots.w * rots.w), // Second element of row 1
		2.f * (rots.z * rots.w - rots.x * rots.y)        // Third element of row 1
	);

	glm::vec3 thirdRow = glm::vec3(
		2.f * (rots.y * rots.w - rots.x * rots.z),       // First element of row 2
		2.f * (rots.z * rots.w + rots.x * rots.y),       // Second element of row 2
		1.f - 2.f * (rots.y * rots.y + rots.z * rots.z)  // Third element of row 2
	);

	glm::mat3 mMatrix = glm::mat3(
		scales.x * glm::vec3(firstRow.x, secondRow.x, thirdRow.x),
		scales.y * glm::vec3(firstRow.y, secondRow.y, thirdRow.y),
		scales.z * glm::vec3(firstRow.z, secondRow.z, thirdRow.z)
	);

	glm::mat3 sigma = glm::transpose(mMatrix) * mMatrix;
	std::vector<float> temp_cov3d;
	temp_cov3d.push_back(sigma[0][0]);
	temp_cov3d.push_back(sigma[0][1]);
	temp_cov3d.push_back(sigma[0][2]);
	temp_cov3d.push_back(sigma[1][1]);
	temp_cov3d.push_back(sigma[1][2]);
	temp_cov3d.push_back(sigma[2][2]);
	
	glm::mat3 t_Vrk = glm::mat3(
		temp_cov3d[0], temp_cov3d[1], temp_cov3d[2],
		temp_cov3d[1], temp_cov3d[2], temp_cov3d[4],
		temp_cov3d[2], temp_cov3d[4], temp_cov3d[5]
	);

	Vrks.push_back(t_Vrk);

	return temp_cov3d;
};

void performPrecalculations(const auto& cloud) {
	normRots.clear();
	for (int i = 0; i < numInstancesCount; ++i) {
		const auto& point = cloud->points[i];
		glm::vec4 rotations = glm::vec4(point.rot_0, point.rot_1, point.rot_2, point.rot_3);
		normRots.push_back(normalizeRotation(rotations));
		expScales.push_back(glm::vec3(exp(point.scale_0), exp(point.scale_1), exp(point.scale_2)));

		const auto& lastRotation = normRots.back();
		const auto& lastScale = expScales.back();
		cov3D.push_back(computeCov3D(lastRotation, lastScale));
	}

	logRotations(normRots);
};


static void computeViewDepths(const auto& splatCloud, const glm::mat4& viewMatrix) {
	viewDepth.clear();
	size_t count = 0;
	for (const auto& point : splatCloud->points) {
		if (count >= numInstancesCount) break;
		glm::vec4 positions = glm::vec4(point.x, point.y, point.z, 1.0f);
		glm::vec4 viewPos = viewMatrix * positions;
		viewDepth.push_back(viewPos.z);
		++count;
	}
	auto maxElement = std::max_element(viewDepth.begin(), viewDepth.end());
	auto minElement = std::min_element(viewDepth.begin(), viewDepth.end());

	std::cout << "Min element" << *minElement << std::endl;
};

std::vector<size_t> radixSort(const auto& splatCloud, const glm::mat4& viewMatrix) {
	const int RADIX_BITS = 8;
	const int RADIX_BUCKETS = 1 << RADIX_BITS;

	computeViewDepths(splatCloud, viewMatrix);

	std::vector<size_t> indices(viewDepth.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<size_t> aux(indices.size());

	for (int shift = 0; shift < 32; shift += RADIX_BITS) {
		std::vector<int> count(RADIX_BUCKETS, 0);

		for (size_t index: indices) {
			int bucket = (*(uint32_t*)&viewDepth[index] >> shift) & (RADIX_BUCKETS - 1);
			count[bucket]++;
		}

		std::vector<int> prefixSum(RADIX_BUCKETS, 0);
		for (int i = 1; i < RADIX_BUCKETS; ++i) {
			prefixSum[i] = prefixSum[i - 1] + count[i - 1];
		}

		for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
			int bucket = (*(uint32_t*)&viewDepth[*it] >> shift) & (RADIX_BUCKETS - 1);
			aux[prefixSum[bucket]++] = *it;
		}
		indices.swap(aux);
	}
	return indices;
};

void performSorting(std::vector<size_t>& sortedIdxArr, const auto& splatCloud) {
	glm::mat4 viewMat = glm::lookAt(
		cameraPos,
		cameraFront,
		cameraUp
	);

	sortedIdxArr = radixSort(splatCloud, viewMat);
};

glm::vec3 SH2RGB(glm::vec3 colors) {
	return 0.5f + C0 * colors;
};

float sigmoid(float opacity) {
	return 1.0 / (1.0 + std::exp(-opacity));
	//return 255.0 / std::exp(-opacity) + 1.0;
}


int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGl", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cout << "Error initializing GLEW" << std::endl;
		return -1;
	}

	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Create a PointCloud with the custom point type
	pcl::PointCloud<GaussianData>::Ptr cloud(new pcl::PointCloud<GaussianData>);

	// Load the PLY file into the cloud
	if (pcl::io::loadPLYFile<GaussianData>("C:/Users/JTSte/Downloads/02880940/02880940-5bb12905529c85359d3d767e1bc88d65.ply", *cloud) == -1) {  // Load PLY file
		std::cerr << "Failed to load .ply file " << std::endl;
		return -1;
	}

	//int numInstances = cloud->points.size();
	int numInstances = std::min(static_cast<size_t>(numInstancesCount), cloud->points.size());

	std::cout << "Loaded " << numInstances << " points " << std::endl;
	std::cout << "Point cloud size " << cloud->points.size() << std::endl;

	// Access and print the scale and rotation values for the first point as a sample
	if (!cloud->points.empty()) {
		GaussianData& point = cloud->points[0];
		std::cout << "First point scale values: "
			<< point.scale_0 << ", " << point.scale_1 << ", " << point.scale_2 << std::endl;
		std::cout << "First point rotation values: "
			<< point.rot_0 << ", " << point.rot_1 << ", " << point.rot_2 << ", " << point.rot_3 << std::endl;
		std::cout << "First point color values: "
			<< point.f_dc_0 << ", " << point.f_dc_0 << ", " << point.f_dc_0 << ", "  << std::endl;
	};

	performPrecalculations(cloud);

	glm::mat4 viewMat = glm::lookAt(
		cameraPos,
		//cameraFront,
		cameraPos + cameraFront,
		cameraUp
	);

	GaussianData& point = cloud->points[0];

	glm::vec4 center = glm::vec4(point.x, point.y, point.z, 1.0f);

	glm::vec4 cam = viewMat * center;

	std::cout << "Cam values " << cam.x << ", " << cam.y << ", " << cam.z << ", " << cam.w << std::endl;

	glm::mat3 m_Vector;

	//m_Vector.push_back(1.0 - 2.0 * rot[0])

	sortedIdx = radixSort(cloud, viewMat);

	std::cout << "View depth sorted idx val " << viewDepth[sortedIdx[0]] << std::endl;

	std::cout << "Sorted idx 0: " << sortedIdx[0] << std::endl;
	std::cout << "Sorted idx 1 " << sortedIdx[1] << std::endl;
	std::cout << "Sorted idx 10: " << sortedIdx[10] << std::endl;
	//std::cout << "Sorted idx 100: " << sortedIdx[100] << std::endl;
	//std::cout << "Sorted idx 1000: " << sortedIdx[1000] << std::endl;

	const char* vertexShaderSource = R"(
		#version 330 core
		layout (location = 0) in vec3 aPos;

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 projection;
		uniform vec3 u_Color;
		uniform float u_Opacity;

		in vec2 triPosition;

		out vec3 fragPos;
		out vec3 outColor;
		//out vec3 normal;
		out float opacity;
		out vec2 vTriPosition;

		vec3 applyQuaternion(vec3 v, vec4 q) {
			return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
		};

		void main() {
			//vec3 scaledPos = instancedScale * aPos;
			//vec3 rotatedPos = applyQuaternion(scaledPos, instancedRotation);
			//vec3 finalPos = rotatedPos + instancedPos;
			gl_Position = projection * view * model * vec4(aPos, 1.0);
			fragPos = aPos;
			outColor = u_Color;
			//normal = applyQuaternion(aPos, instancedRotation);
			opacity = u_Opacity;
			vTriPosition = triPosition;
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 fragPos;
		//in vec3 normal;
		in vec3 outColor;
		in float opacity;
		in vec2 vTriPosition;

		out vec4 FragColor;

		uniform vec3 viewPos;

		void main() {			
			//if(outColor.r < 0 || outColor.g < 0 || outColor.b < 0) discard;

			float A = -dot(vTriPosition, vTriPosition);
			if(A < -4.0) discard;
			float B = exp(A) * opacity;

			//if(opacity < 1. / 255.) discard;

			FragColor = vec4(outColor, B);
			//FragColor = vec4(255.0, 59.0, 130.0, 1.0);
		}
	)";

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach shader source code to shader object & compile shader
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" <<
			infoLog << std::endl;
	}

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" <<
			infoLog << std::endl;
	}

	unsigned int shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	GLfloat triangleVertices[] = {
	-2.0f, -2.0f,
	 2.0f, -2.0f,
	 2.0f,  2.0f,
	-2.0f,  2.0f
	};

	Sphere newSphere;

	unsigned int VBO, VAO, EBO, triangleVBO;
	//glGenBuffers(1, &EBO);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, newSphere.getInterleavedVertexSize(), newSphere.getInterleavedVertices(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, newSphere.getIndexSize(), newSphere.getIndices(), GL_STATIC_DRAW);

	int stride = newSphere.getInterleavedStride();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(float) * 3));
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &triangleVBO);
	glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);

	GLint a_position = glGetAttribLocation(shaderProgram, "triPosition");
	glEnableVertexAttribArray(a_position);
	glVertexAttribPointer(a_position, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);


	//glBindVertexArray(0);
	/*
	std::vector<float> instanceData;
	instanceData.reserve(numInstances * 14); // pos(3) + scale(3) + rot(4) + color(3) + opacity(1)
	//instanceData.reserve(numInstances * 14);

	for (int i = 0; i < numInstances; ++i) {
		const auto& point = cloud->points[sortedIdx[i]];

		// Position
		instanceData.push_back(point.x);
		instanceData.push_back(point.y);
		instanceData.push_back(point.z);
		
		// Scale
		instanceData.push_back(point.scale_0);
		instanceData.push_back(point.scale_1);
		instanceData.push_back(point.scale_2);
		
		// Rotation (quaternion)

		std::vector<glm::vec4> rots = {
			glm::vec4(point.rot_0, point.rot_1, point.rot_2, point.rot_3)
		};

		std::vector<glm::vec4> norms = calculateRotationNorms(rots);

		for (const auto& normVec : norms) {
			instanceData.push_back(normVec.x);
			instanceData.push_back(normVec.y);
			instanceData.push_back(normVec.z);
			instanceData.push_back(normVec.w);
		}

		// Color (from f_dc components)
		instanceData.push_back(point.f_dc_0);
		instanceData.push_back(point.f_dc_1);
		instanceData.push_back(point.f_dc_2);
		
		// Opacity
		instanceData.push_back(point.opacity);
	}


	unsigned int instanceVBO;
	glGenBuffers(1, &instanceVBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glBufferData(GL_ARRAY_BUFFER, numInstances * sizeof(float) * 14, instanceData.data(), GL_STATIC_DRAW);

	// Position
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)0);

	// Scale
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(3);

	// Rotation
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(4);

	// Color 
	glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(10 * sizeof(float)));
	glEnableVertexAttribArray(5);

	// Opacity
	glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(13 * sizeof(float)));
	glEnableVertexAttribArray(6);

	glVertexAttribDivisor(2, 1);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);

	*/

	// Unbind the VBO and VAO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDepthMask(GL_FALSE);

	glEnable(GL_DEPTH_TEST);

	glUseProgram(shaderProgram);

	while (!glfwWindowShouldClose(window)) {

		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window, cloud);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//glm::mat4 model = glm::mat4(1.0f); // Initialize identity matrix
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);

		projection = glm::perspective(glm::radians(fov), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 100.f);
		unsigned int projLoc = glGetUniformLocation(shaderProgram, "projection");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

		float radius = 10.0f;
		float camX = static_cast<float>(sin(glfwGetTime()) * radius);
		float camZ = static_cast<float>(cos(glfwGetTime()) * radius);
		view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

		unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

		float angle = 20.0f;
		//model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));

		for (int i = 0; i < numInstancesCount; ++i) {
			const auto& point = cloud->points[sortedIdx[i]];

			glm::mat4 model = glm::mat4(1.0f);

			model = glm::translate(model, glm::vec3(point.x, point.y, point.z));

			//model = glm::scale(model, glm::vec3(exp(point.scale_0), exp(point.scale_1), exp(point.scale_2)));
			//model = glm::scale(model, glm::vec3(log(point.scale_0), log(point.scale_1), log(point.scale_2)));

			model = glm::scale(model, glm::vec3(expScales[sortedIdx[i]].x, expScales[sortedIdx[i]].y, expScales[sortedIdx[i]].z));

			//glm::quat rotation = glm::quat(point.rot_3, point.rot_0, point.rot_1, point.rot_2);
			//glm::quat rotation = glm::quat(normRots[sortedIdx[i]].w, normRots[sortedIdx[i]].x, normRots[sortedIdx[i]].y, normRots[sortedIdx[i]].z);
			glm::quat rotation = glm::quat(normRots[sortedIdx[i]].x, normRots[sortedIdx[i]].y, normRots[sortedIdx[i]].z, normRots[sortedIdx[i]].w);
			model *= glm::mat4_cast(rotation);

			unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			glm::vec3 colors = SH2RGB(glm::vec3(point.f_dc_0, point.f_dc_1, point.f_dc_2));
			glUniform3f(glGetUniformLocation(shaderProgram, "u_Color"), colors.x, colors.y, colors.z);
			glUniform1f(glGetUniformLocation(shaderProgram, "u_Opacity"), sigmoid(point.opacity));

			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT, (void*)0);
		};
		/*
		unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		*/

		//glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));

		//glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		//glDrawElements(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT,(void*)0);
		//glDrawElementsInstanced(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT, 0, numInstances);
		//glBindVertexArray(0);

		glDepthMask(GL_TRUE);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window, auto& cloud) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	float cameraSpeed = 2.5f * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) *
		cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) *
		cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		/*glm::mat4 viewMat = glm::lookAt(
			cameraPos,
			cameraFront,
			cameraUp
		);
		std::vector<size_t> sortedIdx = radixSort(cloud, viewMat);

		std::cout << "Sorted idx 0: " << sortedIdx[0] << std::endl;
		std::cout << "Sorted idx 1 " << sortedIdx[1] << std::endl;
		std::cout << "Sorted idx 10: " << sortedIdx[10] << std::endl;
		std::cout << "Sorted idx 100: " << sortedIdx[100] << std::endl;
		std::cout << "Sorted idx 1000: " << sortedIdx[1000] << std::endl;
		*/
		
		std::cout << "Camera pos " << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << std::endl;
		std::cout << "Camera Front " << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z << std::endl;
		std::cout << "Camera up" << cameraUp.x << ", " << cameraUp.y << ", " << cameraUp.z << std::endl;
		

		performSorting(sortedIdx, cloud);
	}
}
//
///* Resizes the viewpoint when the window is resized */
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	//
	//    /* Tells OpenGL the size of the rendering window so it can display data & coords w.r.t the window */
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	fov -= (float)yoffset;
	if (fov < 1.0f)
		fov = 1.0f;
	if (fov > 45.0f)
		fov = 45.0f;
}
