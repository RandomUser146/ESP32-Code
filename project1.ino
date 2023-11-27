#include "model5_low.h"
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "vector"
#define RXp2 16
#define TXp2 17
#define DEBUG 1

// constants
constexpr float r_training_mean = 0.28758077777777774;
constexpr float r_training_std = 0.7389764066504841;
constexpr float h_training_mean = 84.14341222222222;
constexpr float h_training_std = 13.33871610684333;
constexpr float t_training_mean = 25.96060955555557;
constexpr float t_training_std = 2.5663260539356045;
constexpr float p_training_mean = 1008.424936;
constexpr float p_training_std = 2.028680341807781;

float x[9][8] = { { 9.63854967e-01, 1.09938452e+00, 8.25691444e-01,
                    -1.15363734e+00, -5.94759545e-12, 1.00000000e+00,
                    -9.26932319e-01, -3.75228564e-01 },
                  { 8.88885233e-01, 9.64062203e-01, 5.29932675e-01,
                    -1.15363734e+00, 2.58819045e-01, 9.65925826e-01,
                    -9.27201038e-01, -3.74564059e-01 },
                  { 8.88885233e-01, 9.64062203e-01, 1.35587650e-01,
                    -1.19260355e+00, 5.00000000e-01, 8.66025404e-01,
                    -9.27469281e-01, -3.73899362e-01 },
                  { 9.63854967e-01, 1.23470684e+00, 2.34173906e-01,
                    -1.23156976e+00, 7.07106781e-01, 7.07106781e-01,
                    -9.27737047e-01, -3.73234472e-01 },
                  { 1.03882470e+00, 1.23470684e+00, 1.84880778e-01,
                    -1.23156976e+00, 8.66025404e-01, 5.00000000e-01,
                    -9.28004337e-01, -3.72569391e-01 },
                  { 1.03882470e+00, 1.23470684e+00, 3.32760162e-01,
                    -1.19260355e+00, 9.65925826e-01, 2.58819045e-01,
                    -9.28271150e-01, -3.71904118e-01 },
                  { 1.03882470e+00, 1.50535147e+00, 4.31346419e-01,
                    -1.23156976e+00, 1.00000000e+00, 8.51093960e-12,
                    -9.28537485e-01, -3.71238654e-01 },
                  { 1.03882470e+00, 1.23470684e+00, 7.27105187e-01,
                    -1.15363734e+00, 9.65925826e-01, -2.58819045e-01,
                    -9.28803344e-01, -3.70572999e-01 },
                  { 9.63854967e-01, 9.64062203e-01, 1.02286396e+00,
                    -9.97772497e-01, 8.66025404e-01, -5.00000000e-01,
                    -9.29068726e-01, -3.69907154e-01 } };
float y = 0.8287398849417935;
float y_predicted = 1.045043;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(9600);
  Serial2.begin(9600, SERIAL_8N1, RXp2, TXp2);
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model5);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(input->dims->data[2]);
  Serial.print("Input type: ");
  Serial.println(input->type);
#endif
}

// The name of this function is important for Arduino compatibility.
void loop() {
  Serial.println("Message Received: ");
  Serial.println(Serial2.readString());
  // Run inference, and report any error
  delay(1000);
  int input_ = interpreter->inputs()[0];
  float* input_data_ptr = interpreter->typed_input_tensor<float>(input_);
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 8; j++) {
      *(input_data_ptr) = (float)x[i][j];
      input_data_ptr++;
    }
  }
  TfLiteStatus invoke_status = interpreter->Invoke();
  float o = output->data.f[0];
  Serial.printf("Value model outputted : %f", o);
  Serial.printf("Value expected : %f", y);
  Serial.printf("Value tflite python predicted : %f", y_predicted);
  Serial.println("OK!! at last!............");
  if (invoke_status != kTfLiteOk) {
    return;
  }
}