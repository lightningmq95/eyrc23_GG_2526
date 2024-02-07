#include <WiFi.h>

// WiFi credentials
const char* ssid = "LIGHTNINGMQ95";
const char* password = "AkkiMohapatra@3010";
const uint16_t port = 8002;
const char* host = "192.168.137.1";

// Motor pin connections
#define in1 27
#define in2 14
#define in3 12
#define in4 13

// IR sensor pin connections
#define R 23   //D23
#define ML 21  //D21
#define M 22   //D22
#define MR 17  //TX2
#define L 18   //D18

#define ledPin 2      // Change to the appropriate pin number
#define buzzerPin 15  // Change to the appropriate pin number

int counter = 0;
bool buzzerActivated = false;
unsigned long startMillis;
unsigned long currentMillis;

WiFiServer server(port);
WiFiClient client;

void setup() {
  Serial.begin(9600);

  // Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // IR sensor pins as input
  pinMode(L, INPUT);
  pinMode(ML, INPUT);
  pinMode(M, INPUT);
  pinMode(MR, INPUT);
  pinMode(R, INPUT);

  // buzzer and led
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  // initially buzzer off
  digitalWrite(buzzerPin, HIGH);

  // Keep all motors off initially
  stopMotors();

  // Connecting to WiFi
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

  // Start the server
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
    if (command == '1') {
      stopMotors();
      activateBL();
    } if (command == '0') {
      deactivateBL();
    } if (command == '2') {
      moveForward();
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
  // half working
  if (digitalRead(L) == LOW && digitalRead(ML) == HIGH && digitalRead(M) == HIGH && digitalRead(MR) == HIGH && digitalRead(R) == LOW) {
    if (!buzzerActivated) {
      buzzerActivated = true;
      currentMillis = millis();
      while (millis() - currentMillis < 1000) {
        stopMotors();
        digitalWrite(buzzerPin, LOW);
      }
      digitalWrite(buzzerPin, HIGH);  // Turn off the buzzer
      counter++;
      client.println(counter);
      node_count(counter);
    }
  } else {
    buzzerActivated = false;
  }
  // // working
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

void node_count(int counter) {
  if (counter == 1) {
    moveForward();
  }
  if (counter == 2) {
    moveForward();
  }
  if (counter == 3) {
    litForward();
    while (digitalRead(R) != HIGH) {
      pivot_right();
    }
    while (digitalRead(M) != HIGH) {
      pivot_right();
    }
  }
  if (counter == 4) {
    litlessForward();
    while (digitalRead(L) != HIGH) {
      pivot_left();
    }
    while (digitalRead(M) != HIGH) {
      pivot_left();
    }
  }
  if (counter == 5) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    delay(250);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
    delay(100);
    while (digitalRead(R) != HIGH) {
      pivot_right();
    }
    while (digitalRead(M) != HIGH) {
      pivot_right();
    }
  }
  if (counter == 6) {
    litForward();
    while (digitalRead(R) != HIGH) {
      pivot_right();
    }
    while (digitalRead(M) != HIGH) {
      pivot_right();
    }
  }
  if (counter == 7) {
    moveForward();
  }
  if (counter == 8) {
    litForward();
    while (digitalRead(R) != HIGH) {
      pivot_right();
    }
    while (digitalRead(M) != HIGH) {
      pivot_right();
    }
  }
  if (counter == 9) {
    moveForward();
  }
  if (counter == 10) {
    litForward();
    while (digitalRead(L) != HIGH) {
      pivot_left();
    }
    while (digitalRead(M) != HIGH) {
      pivot_left();
    }
  }
  if (counter == 11) {
    moveForward();
  }
}

void activateBL() {
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, LOW);
  delay(1000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, HIGH);
}

void deactivateBL() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, LOW);
  delay(5000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, HIGH);
}

void moveForward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void litForward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(450);
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
  delay(300);
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
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