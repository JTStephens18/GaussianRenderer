// GaussianRenderer.cpp : Defines the entry point for the application.

#include "GaussianRenderer.h"
#include "./Sphere.h"
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
//void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
//void cursor_position_callback(GLFWwindow* window, double xpos, double ypos); 

const unsigned int SCREEN_WIDTH = 800;
const unsigned int SCREEN_HEIGHT = 600;

glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.f;
float pitch = 0.0f;
float lastX = SCREEN_WIDTH / 2.0;
float lastY = SCREEN_HEIGHT / 2.0;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

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
	if (pcl::io::loadPLYFile<GaussianData>("C:/Users/JTSte/Downloads/02880940/02880940-817221e45b63cef62f74bdafe5239fba.ply", *cloud) == -1) {  // Load PLY file
		std::cerr << "Failed to load .ply file " << std::endl;
		return -1;
	}

	//int numInstances = cloud->points.size();
	int numInstances = std::min(static_cast<size_t>(1000), cloud->points.size());

	std::cout << "Loaded " << numInstances << " points " << std::endl;

	// Access and print the scale and rotation values for the first point as a sample
	if (!cloud->points.empty()) {
		GaussianData& point = cloud->points[0];
		std::cout << "First point scale values: "
			<< point.scale_0 << ", " << point.scale_1 << ", " << point.scale_2 << std::endl;
		std::cout << "First point rotation values: "
			<< point.rot_0 << ", " << point.rot_1 << ", " << point.rot_2 << ", " << point.rot_3 << std::endl;
	};

	const char* vertexShaderSource = R"(
		#version 330 core
		layout (location = 0) in vec3 aPos;

		layout (location = 2) in vec3 instancedPos;
		layout (location = 3) in vec3 instancedScale;
		layout (location = 4) in vec4 instancedRotation;
		layout (location = 5) in vec3 instancedColor;
		layout (location = 6) in float instancedOpacity;

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 projection;	

		out vec3 fragPos;
		out vec3 outColor;
		out vec3 normal;
		out float opacity;

		vec3 applyQuaternion(vec3 v, vec4 q) {
			return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
		};

		void main() {
			vec3 scaledPos = instancedScale * aPos;
			vec3 rotatedPos = applyQuaternion(scaledPos, instancedRotation);
			vec3 finalPos = rotatedPos + instancedPos;
			gl_Position = projection * view * model * vec4(finalPos, 1.0);
			fragPos = finalPos;
			outColor = instancedColor;
			normal = applyQuaternion(aPos, instancedRotation);
			opacity = instancedOpacity;
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 fragPos;
		in vec3 normal;
		in vec3 outColor;
		in float opacity;

		out vec4 FragColor;

		uniform vec3 viewPos;

		void main() {
			vec3 lightPos = vec3(5.0f, 5.0f, 5.0f);
			vec3 norm = normalize(normal);
			vec3 lightDir = normalize(lightPos - fragPos);
			//vec3 viewDir = normalize(viewPos - fragPos);

			float diff = max(dot(norm, lightDir), 0.0);
			vec3 diffuse = diff * outColor;

			vec3 ambient = 0.1 * outColor;

			FragColor = vec4(ambient + diffuse, opacity);
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

	Sphere newSphere;

	unsigned int VBO, VAO, EBO;
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


	glBindVertexArray(0);


	std::vector<float> instanceData;
	instanceData.reserve(numInstances * 14); // pos(3) + scale(3) + rot(4) + color(3) + opacity(1)
	//instanceData.reserve(numInstances * 14);

	for (int i = 0; i < numInstances; ++i) {
		const auto& point = cloud->points[i];

		// Position
		instanceData.push_back(point.x);
		instanceData.push_back(point.y);
		instanceData.push_back(point.z);
		
		// Scale
		instanceData.push_back(point.scale_0);
		instanceData.push_back(point.scale_1);
		instanceData.push_back(point.scale_2);
		
		// Rotation (quaternion)
		instanceData.push_back(point.rot_0);
		instanceData.push_back(point.rot_1);
		instanceData.push_back(point.rot_2);
		instanceData.push_back(point.rot_3);

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

	// Unbind the VBO and VAO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glUseProgram(shaderProgram);

	while (!glfwWindowShouldClose(window)) {

		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 model = glm::mat4(1.0f); // Initialize identity matrix
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
		model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));

		unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		//glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		//glDrawElements(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT,(void*)0);
		glDrawElementsInstanced(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT, 0, numInstances);
		glBindVertexArray(0);

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

void processInput(GLFWwindow* window) {
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
