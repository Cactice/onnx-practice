// Language: typescript
// Path: react-next\utils\predict.ts
import { preprocessed } from "../data/preprocessed"
import * as ort from 'onnxruntime-web';
import { NextPage } from "next";

export async function runWhisperModel(): Promise<[any, number]> {
  const session = await ort.InferenceSession
    .create('./_next/static/chunks/pages/encoder_model.onnx',
      { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
  const [results, inferenceTime] = await runInference(session);
  console.log([results, inferenceTime])
  return [results, inferenceTime];
}

async function runInference(session: ort.InferenceSession): Promise<[any, number]> {

  const preprocessedData = new ort.Tensor('float32', preprocessed.flat().flat(), [1, 80, 3000]);
  // Get start time to calculate inference time.
  const start = new Date();
  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {
    [session.inputNames[0]]: preprocessedData
  };
  console.log(session.inputNames[0])

  // Run the session inference.
  const outputData = await session.run(feeds);
  // Get the end time to calculate inference time.
  const end = new Date();
  // Convert to seconds.
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];
  return [output, inferenceTime];
}

const Home: NextPage = () => {

  return (
    <>
      <button
        onClick={runWhisperModel} >
        Run whisper inference
      </button>
      <br />
    </>
  )
}

export default Home
