#include <WiFi.h>

// WiFi credentials
const char* ssid = "JioFiber-Ynqef_5G";
const char* password = "sohamguj@1234";
const uint16_t port = 8002;
const char * host = "192.168.29.191";

// Motor pin connections
#define in1 16
#define in2 4
#define in3 2
#define in4 15

WiFiServer server(port);

void setup() {
 Serial.begin(9600);

 // Motor pins as output
 pinMode(in1, OUTPUT);
 pinMode(in2, OUTPUT);
 pinMode(in3, OUTPUT);
 pinMode(in4, OUTPUT);

 // Connecting to WiFi
 WiFi.begin(ssid, password);

 while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
 }

 Serial.print("WiFi connected with IP: ");
 Serial.println(WiFi.localIP());

 // Start the server
 server.begin();
}

WiFiClient client;

void loop() {
  // Check if a client is connected
  if (!client || !client.connected()) {
    // Wait for a new client to connect
    client = server.available();
    // delay(1000); // Wait for 1 second before checking again
    return;
  }

  // Read the command from the client
  String command = client.readStringUntil('\n');

  // If the command is not empty, process it
  if (command.length() > 0) {
    Serial.println("Received command: " + command);

    // Process the command and control the motor
    if (command.toInt() == 1) {
      // Move
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    } else if (command.toInt() == 0) {
      // Stop
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
    }

    // Send a response back to the client
    client.print("Command received: ");
    client.println(command);
  }
}
