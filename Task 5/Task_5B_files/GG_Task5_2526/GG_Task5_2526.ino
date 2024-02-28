#include <WiFi.h>

const char* ssid = "LIGHTNINGMQ95";
const char* password = "AkkiMohapatra@3010";
const uint16_t port = 8002;
const char* host = "192.168.137.1";

#define in1 27
#define in2 14
#define in3 12
#define in4 13

#define R 23   //D23
#define ML 21  //D21
#define M 22   //D22
#define MR 17  //TX2
#define L 18   //D18

#define ledPin 2
#define buzzerPin 15

unsigned long startMillis;
unsigned long currentMillis;

WiFiServer server(port);
WiFiClient client;


void setup() {
  Serial.begin(9600);

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  pinMode(L, INPUT);
  pinMode(ML, INPUT);
  pinMode(M, INPUT);
  pinMode(MR, INPUT);
  pinMode(R, INPUT);

  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  digitalWrite(buzzerPin, HIGH);

  stopMotors();

  Serial.print("Connecting to: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
    attempts++;
    if (attempts > 3) {
      ESP.restart();
    }
  }
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

  server.begin();

  startMillis = millis();
}


void loop() {
  // Check if a Serial is connected
  if (!client || !client.connected()) {
    // Wait for a new Serial to connect
    client = server.available();
    return;
  }

  if (client.available()) {
    char command = client.read();
    client.println(command);
    if (command == '1') {
      moveForward();
    }

    if (command == '2') {
      currentMillis = millis();
      while (millis() - currentMillis < 490) {
        pivot_left();
      }
    }

    if (command == '3') {
      currentMillis = millis();
      while (millis() - currentMillis < 490) {
        pivot_right();
      }
    }

    if (command == '4') {
      currentMillis = millis();
      while (millis() - currentMillis < 1050) {
        pivot_left();
      }
    }

    if (command == '5') {
      deactivateBL();
    }

    if (command == '0') {
      currentMillis = millis();
      while (millis() - currentMillis < 1000) {
        stopMotors();
        digitalWrite(buzzerPin, LOW);
      }
      digitalWrite(buzzerPin, HIGH);
    }
  }

  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == LOW && digitalRead(M) == HIGH && digitalRead(MR) == LOW && digitalRead(R) == LOW) {
    moveForward();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == HIGH && digitalRead(M) == LOW && digitalRead(MR) == LOW && digitalRead(R) == LOW) {
    turnLeft();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == HIGH && digitalRead(M) == HIGH && digitalRead(MR) == LOW && digitalRead(R) == LOW) {
    turnLeft();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == LOW && digitalRead(M) == LOW && digitalRead(MR) == HIGH && digitalRead(R) == LOW) {
    turnRight();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == LOW && digitalRead(M) == HIGH && digitalRead(MR) == HIGH && digitalRead(R) == LOW) {
    turnRight();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(ML) == HIGH && digitalRead(M) == HIGH && digitalRead(MR) == HIGH && digitalRead(R) == LOW) {
    litlessForward();
    client.println("9");
  }
  // working
  if (digitalRead(L) == HIGH && digitalRead(ML) == LOW && digitalRead(M) == HIGH && digitalRead(MR) == LOW && digitalRead(R) == HIGH) {
    moveForward();
  }
  //working
  if (digitalRead(L) == LOW && digitalRead(ML) == LOW && digitalRead(M) == HIGH && digitalRead(MR) == LOW && digitalRead(R) == HIGH) {
    turnLeft();
  }
  //working
  if (digitalRead(L) == HIGH && digitalRead(ML) == LOW && digitalRead(M) == HIGH && digitalRead(MR) == LOW && digitalRead(R) == LOW) {
    turnRight();
  }
  //working
  if (digitalRead(L) == HIGH && digitalRead(ML) == LOW && digitalRead(M) == LOW && digitalRead(MR) == LOW && digitalRead(R) == HIGH) {
    moveForward();
  }
  //working
  if (digitalRead(L) == LOW && digitalRead(ML) == LOW && digitalRead(M) == LOW && digitalRead(MR) == LOW && digitalRead(R) == HIGH) {
    turnLeft();
  }
  //working
  if (digitalRead(L) == HIGH && digitalRead(ML) == LOW && digitalRead(M) == LOW && digitalRead(MR) == LOW && digitalRead(R) == LOW) {
    turnRight();
  }
  //working
  if (digitalRead(L) == HIGH && digitalRead(ML) == HIGH && digitalRead(M) == HIGH && digitalRead(MR) == HIGH && digitalRead(R) == HIGH) {
    moveForward();
  }
  // working
  if (digitalRead(L) == LOW && digitalRead(R) == LOW && digitalRead(M) == LOW && digitalRead(MR) == LOW && digitalRead(ML) == LOW) {
    moveForward();
  }
}


void moveForward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}


void turnRight() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}



void pivot_right() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}



void turnLeft() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}


void pivot_left() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}


void stopMotors() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}


void litlessForward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(290);
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}


void deactivateBL() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(buzzerPin, LOW);
  delay(5000);
  digitalWrite(buzzerPin, HIGH);
  currentMillis = millis();
  while (millis() - currentMillis < 1000) {
    stopMotors();
  }
}