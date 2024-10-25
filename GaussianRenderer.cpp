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

	const char* vertexShaderSource = R"(
		#version 330 core
		layout (location = 0) in vec3 aPos;

		out vec3 outColor;

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 projection;	

		uniform vec3 center;
		uniform vec3 scale;
		uniform vec4 rotation;

		out vec3 fragPos;

		vec3 applyQuaternion(vec3 v, vec4 q) {
			return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
		};

		void main() {
			vec3 scaledPos = scale * aPos;
			vec3 rotatedPos = applyQuaternion(scaledPos, rotation);
			vec3 finalPos = rotatedPos + center;
			gl_Position = projection * view * model * vec4(finalPos, 1.0);
			fragPos = scaledPos;
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 fragPos;
		out vec4 FragColor;

		uniform float opacity;
		uniform vec3 color;
		uniform vec3 spHarmonics;

		void main() {
			float gaussianValue = exp(-dot(fragPos, fragPos));
			vec3 harmonicsColor = color + spHarmonics * gaussianValue;
			FragColor = vec4(harmonicsColor, opacity * gaussianValue);
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

	unsigned int shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// Temp gaussian parameters
	glm::vec3 center = glm::vec3(-0.20695215f, -0.07309745, -0.032607663);
	glm::vec3 scales = glm::vec3(-4.38290596f, -5.57705736f, -5.53777361f);
	glm::quat rotation = glm::quat(0.93376312f, 0.02444352f, 0.3460566f, 0.08794192f);
	glm::vec3 spHarmonic = glm::vec3(0.25771704f, 0.25211358f, 0.24187477f);
	glm::vec3 color = glm::vec3(1.0f, 0.2f, 0.5f);
	float opacity = -1.3434249f;

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

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	int stride = newSphere.getInterleavedStride();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(float) * 3));

	// Unbind the VBO and VAO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

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

		unsigned int centerLoc = glGetUniformLocation(shaderProgram, "center");
		glUniform3fv(centerLoc, 1, glm::value_ptr(center));

		unsigned int scaleLoc = glGetUniformLocation(shaderProgram, "scale");
		glUniform3fv(scaleLoc, 1, glm::value_ptr(scales));

		unsigned int rotLoc = glGetUniformLocation(shaderProgram, "rotation");
		glUniform4fv(rotLoc, 1, glm::value_ptr(rotation));

		unsigned int opacityLoc = glGetUniformLocation(shaderProgram, "opacity");
		glUniform1f(opacityLoc, opacity);

		unsigned int colorLoc = glGetUniformLocation(shaderProgram, "color");
		glUniform3fv(colorLoc, 1, glm::value_ptr(color));

		unsigned int sphLoc = glGetUniformLocation(shaderProgram, "spHarmonics");
		glUniform3fv(sphLoc, 1, glm::value_ptr(spHarmonic));

		glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glDrawElements(GL_TRIANGLES, newSphere.getIndexCount(), GL_UNSIGNED_INT, (void*)0);
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
