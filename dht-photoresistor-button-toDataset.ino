//PHOTORESISTOR
// S -> A0
// + -> 3.3 V
// - -> gnd

//BUTTON
//out -> D4
//gnd -> gnd
// vcc -> 3.3 V

//DHT
// s -> D1
// + -> 3.3 V
// - -> gnd


#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h>
String readString;
/*
 * Rui Santos 
 * Complete Project Details https://randomnerdtutorials.com
 */
 
#ifdef ESP32
  #include <WiFi.h>
#else
  #include <ESP8266WiFi.h>
#endif

#include <Wire.h>
#include <Adafruit_Sensor.h>

#include "DHT.h"        // including the library of DHT11 temperature and humidity sensor
#define DHTTYPE DHT11   // DHT 11

#define dht_dpin 5
DHT dht(dht_dpin, DHTTYPE); 

// Replace with your SSID and Password
const char* ssid     = "DavidsonGuest";
const char* password = "";

// Replace with your unique IFTTT URL resource
const char* resource = "https://maker.ifttt.com/trigger/esp8266_readings/with/key/-FRuzwR_UK1y1hqIgN7U2H1WphQaV5vN5oiHCvXRVn";

// How your resource variable should look like, but with your own API KEY (that API KEY below is just an example):
//const char* resource = "/trigger/bme280_readings/with/key/nAZjOphL3d-ZO4N3k64-1A7gTlNSrxMJdmqy3";

// Maker Webhooks IFTTT
const char* server = "maker.ifttt.com";

// Time to sleep
uint64_t uS_TO_S_FACTOR = 1000000;  // Conversion factor for micro seconds to seconds
// sleep for 30 minutes = 1800 seconds
uint64_t TIME_TO_SLEEP = 1800;

const int buttonPin = 2;
const int ledPin =  13; 
int buttonState = 0;
int prevState= 0;



// Uncomment to use BME280 SPI
/*#include <SPI.h>
#define BME_SCK 13
#define BME_MISO 12
#define BME_MOSI 11
#define BME_CS 10*/

#define SEALEVELPRESSURE_HPA (1013.25)
// I2C
//Adafruit_BME280 bme(BME_CS); // hardware SPI
//Adafruit_BME280 bme(BME_CS, BME_MOSI, BME_MISO, BME_SCK); // software SPI

void setup() {
  dht.begin();
  Serial.begin(115200); 
  initWifi();
  pinMode(buttonPin, INPUT);
}

void loop() {
    // read the state of the pushbutton value:
  buttonState = digitalRead(buttonPin);

  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
    //if button state has changed, print button state and make HTTP request
    makeIFTTTRequest();
    Serial.println(buttonState);
    float h = dht.readHumidity();
    float t = dht.readTemperature();         
    Serial.print("Current humidity = ");
    Serial.print(h);
    Serial.print("%  ");
    Serial.print("temperature = ");
    Serial.print(t); 
    Serial.println("C  ");
    int sensorValue = analogRead(A0);
    Serial.println(sensorValue);
    delay(60000);
}

// Establish a Wi-Fi connection with your router
void initWifi() {
  Serial.print("Connecting to: "); 
  Serial.print(ssid);
  WiFi.begin(ssid, password);  

  int timeout = 10 * 4; // 10 seconds
  while(WiFi.status() != WL_CONNECTED  && (timeout-- > 0)) {
    delay(250);
    Serial.print(".");
  }
  Serial.println("");

  if(WiFi.status() != WL_CONNECTED) {
     Serial.println("Failed to connect, going back to sleep");
  }

  Serial.print("WiFi connected in: "); 
  Serial.print(millis());
  Serial.print(", IP address: "); 
  Serial.println(WiFi.localIP());
}

// Make an HTTP request to the IFTTT web service
void makeIFTTTRequest() {
  Serial.print("Connecting to "); 
  Serial.print(server);
  
  WiFiClient client;
  int retries = 5;
  while(!!!client.connect(server, 80) && (retries-- > 0)) {
    Serial.print(".");
  }
  Serial.println();
  if(!!!client.connected()) {
    Serial.println("Failed to connect...");
  }
  
  Serial.print("Request resource: "); 
  Serial.println(resource);

  // Building data to send to google Sheets
 String jsonObject = String("{\"value1\":\"") + String(buttonState) + "\",\"value2\":\"" + String(analogRead(A0))
                      + "\",\"value3\":\"" + String(dht.readTemperature())+";"+ String(dht.readHumidity()) + "\"}";
                 
 
  client.println(String("POST ") + resource + " HTTP/1.1");
  client.println(String("Host: ") + server); 
  client.println("Connection: close\r\nContent-Type: application/json");
  client.print("Content-Length: ");
  client.println(jsonObject.length());
  client.println();
  client.println(jsonObject);
        
  int timeout = 5 * 10; // 5 seconds             
  while(!!!client.available() && (timeout-- > 0)){
    delay(100);
  }
  if(!!!client.available()) {
    Serial.println("No response...");
  }
  while(client.available()){
    Serial.write(client.read());
  }
  
  Serial.println("\nclosing connection");
  client.stop(); 
}
