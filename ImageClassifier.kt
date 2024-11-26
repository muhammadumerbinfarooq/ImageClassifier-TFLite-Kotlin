import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.TensorBuffer
import org.tensorflow.lite.support.common.FileUtil

class ImageClassifier(private val activity: Activity) {
    private lateinit var interpreter: Interpreter
    private lateinit var inputTensor: Tensor
    private lateinit var outputTensor: Tensor

    fun initModel() {
        // Load the TensorFlow Lite model
        val model = FileUtil.loadMappedFile(activity, "model.tflite")
        interpreter = Interpreter(model)
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)
    }

    fun classifyImage(bitmap: Bitmap): String {
        // Preprocess the image
        val inputBuffer = TensorBuffer.createFixedSize(inputTensor.shape(), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputBuffer.capacity())
        byteBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.loadArray(byteBuffer.asFloatBuffer())

        // Run the inference
        interpreter.run(inputBuffer.buffer, outputTensor.buffer)

        // Get the output
        val outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), DataType.FLOAT32)
        outputTensor.copyTo(outputBuffer.buffer)

        // Postprocess the output
        val probabilities = outputBuffer.floatArray
        val maxIndex = probabilities.indexOf(probabilities.maxOrNull()!!)
        return when (maxIndex) {
            0 -> "Cat"
            1 -> "Dog"
            else -> "Unknown"
        }
    }
}
