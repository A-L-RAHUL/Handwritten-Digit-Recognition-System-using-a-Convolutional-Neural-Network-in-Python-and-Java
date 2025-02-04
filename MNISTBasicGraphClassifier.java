package org.tensorflow.keras.examples.mnist;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.data.Dataset;
import org.tensorflow.data.DatasetIterator;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.training.optimizers.GradientDescent;
import org.tensorflow.training.optimizers.Optimizer;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import java.io.IOException;
import java.util.List;
import static org.tensorflow.tools.ndarray.NdArrays.vectorOf;
/**
* An example showing a simple feed-forward classifier for MNIST
* using tf.data and core TensorFlow (in Graph Mode).
*/
public class MNISTBasicGraphClassifier implements Runnable {

private static final int INPUT_SIZE = 28 * 28;
private static final float LEARNING_RATE = 0.2f;
private static final int FEATURES = 10;
private static final int BATCH_SIZE = 100;
private static final int EPOCHS = 10;
static class EpochLogs {
float loss = 0;
float accuracy = 0;
int batches = 0;
void batchUpdate(float batchLoss, float batchAccuracy) {
++batches;
loss += batchLoss;
accuracy += batchAccuracy;
}
}
public static void main(String[] args) {
new MNISTBasicGraphClassifier().run();
}
public Operand<TFloat32> predict(Ops tf, Operand<TFloat32> images,
Variable<TFloat32> weights,

Variable<TFloat32> biases) {
return tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights),
biases));
}
public Operand<TFloat32> crossEntropyLoss(Ops tf, Operand<TFloat32>
predicted, Operand<TFloat32> actual) {
return tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(actual,
tf.math.log(predicted)), tf.constant(1))),
tf.constant(0));
}
public Operand<TFloat32> accuracy(Ops tf, Operand<TFloat32> predicted,
Operand<TFloat32> actual) {
Operand<TInt64> yHat = tf.math.argMax(predicted, tf.constant(1));
Operand<TInt64> yTrue = tf.math.argMax(actual, tf.constant(1));
Operand<TFloat32> accuracy =
tf.math.mean(tf.dtypes.cast(tf.math.equal(yHat, yTrue), TFloat32.DTYPE),
tf.constant(0));
return accuracy;
}

public void run() {
try (Graph graph = new Graph()) {
Ops tf = Ops.create(graph);
// Load datasets
MNIST mnist;
try {
mnist = MNIST.loadData(tf);
} catch (IOException e) {
System.out.println("Could not load dataset.");
return;
}
Dataset train = mnist.train.batch(BATCH_SIZE);
Dataset test = mnist.test.batch(BATCH_SIZE);
// Extract iterators and train / test components
DatasetIterator trainIterator = train.makeInitializeableIterator();
Op initTrainIterator = trainIterator.makeInitializer(train);
List<Output<?>> components = trainIterator.getNext();
Operand<TFloat32> trainImages =
components.get(0).expect(TFloat32.DTYPE);
Operand<TFloat32> trainLabels =
components.get(1).expect(TFloat32.DTYPE);
DatasetIterator testIterator = test.makeInitializeableIterator();
Op initTestIterator = testIterator.makeInitializer(test);
List<Output<?>> testComponents = testIterator.getNext();
Operand<TFloat32> testImages =
testComponents.get(0).expect(TFloat32.DTYPE);
Operand<TFloat32> testLabels =
testComponents.get(1).expect(TFloat32.DTYPE);

// Flatten image tensors
trainImages = tf.reshape(trainImages,
tf.constant(vectorOf(-1, INPUT_SIZE)));
testImages = tf.reshape(testImages,
tf.constant(vectorOf(-1, INPUT_SIZE)));
// Declare, initialize weights
Variable<TFloat32> weights = tf.variable(Shape.of(INPUT_SIZE,

FEATURES), TFloat32.DTYPE);
Assign<TFloat32> weightsInit = tf.assign(weights,
tf.zeros(tf.constant(vectorOf(INPUT_SIZE, FEATURES)),

TFloat32.DTYPE));
Variable<TFloat32> biases = tf.variable(Shape.of(FEATURES),
TFloat32.DTYPE);
Assign<TFloat32> biasesInit = tf.assign(biases,
tf.zeros(tf.constant(vectorOf(FEATURES)), TFloat32.DTYPE));
// SETUP: Training
Operand<TFloat32> trainPrediction = predict(tf, trainImages, weights,
biases);
Operand<TFloat32> trainAccuracy = accuracy(tf, trainPrediction,
trainLabels);
Operand<TFloat32> trainLoss = crossEntropyLoss(tf, trainPrediction,
trainLabels);
Optimizer gradientDescent = new GradientDescent(graph, LEARNING_RATE);
Op optimizerTargets = gradientDescent.minimize(trainLoss);
// SETUP: Testing
Operand<TFloat32> testPrediction = predict(tf, testImages, weights,
biases);
Operand<TFloat32> testAccuracy = accuracy(tf, testPrediction,
testLabels);
Operand<TFloat32> testLoss = crossEntropyLoss(tf, testPrediction,
testLabels);
try (Session session = new Session(graph)) {
// Initialize weights and biases
session.runner()
.addTarget(weightsInit)
.addTarget(biasesInit)
.run();
// Run training loop
for (int i = 0; i < EPOCHS; i++) {
// reset iterator object
session.run(initTrainIterator);
EpochLogs epochLogs = new EpochLogs();
session.runner()

.addTarget(optimizerTargets)
.fetch(trainLoss)
.fetch(trainAccuracy)
.repeat()
.limit(500)
.forEach(outputs ->
epochLogs.batchUpdate(outputs.popFloat(),

outputs.popFloat())
);
System.out.println("Epoch Accuracy " + i + ": " +

epochLogs.accuracy / epochLogs.batches);
}
// Evaluate on test set
session.run(initTestIterator);
EpochLogs testLogs = new EpochLogs();
session.runner()
.fetch(testLoss)
.fetch(testAccuracy)
.repeat()
.forEach(outputs -> testLogs.accuracy += outputs.popFloat());
System.out.println("Test Accuracy " + ": " + testLogs.accuracy);
}
}
}
}
