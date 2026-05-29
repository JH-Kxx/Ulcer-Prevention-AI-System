const int s0 = A0;
const int s1 = A1;
const int s2 = A2;
const int s3 = A3;

const int s4 = A4;
const int s5 = A5;
const int s6 = A6;
const int s7 = A7;

void setup() {
  Serial.begin(115200);
}

void loop() {

  Serial.print(analogRead(s0));
  Serial.print(",");

  Serial.print(analogRead(s1));
  Serial.print(",");

  Serial.print(analogRead(s2));
  Serial.print(",");

  Serial.print(analogRead(s3));
  Serial.print(",");

  Serial.print(analogRead(s4));
  Serial.print(",");

  Serial.print(analogRead(s5));
  Serial.print(",");

  Serial.print(analogRead(s6));
  Serial.print(",");

  Serial.println(analogRead(s7));

  delay(30);
}