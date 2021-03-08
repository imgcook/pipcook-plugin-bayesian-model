/**
 * @file This is for the plugin to load Bayes Classifier model.
 */

import { UniModel, CsvDataset } from '@pipcook/pipcook-core';
import * as assert from 'assert';
import * as path from 'path';
import * as fs from 'fs-extra';
import {
  getBayesModel,
  loadModel,
  TextProcessing,
  MakeWordsSet,
  words_dict,
  TextFeatures,
  save_all_words_list,
  saveBayesModel
} from './script';
import { cn, en } from './stopwords';

/**
 * Pipcook Plugin: bayes classifier model
 * @param data Pipcook uniform sample data
 * @param args args. If the model path is provided, it will restore the model previously saved
 */
const modelDefine = async (options: Record<string, any>, api: any): Promise<any> => {
  const { boa } = api;
  const sys = boa.import('sys');
  const {
    recoverPath
  } = options;
      
  sys.path.insert(0, path.join(__dirname, 'assets'));
  let classifier: any;

  if (!recoverPath) {
    // assertionTest(data);
    classifier = getBayesModel(boa);
  } else {
    classifier = await loadModel(path.join(recoverPath, 'model.pkl'), boa);
  }
  return classifier;
};

/**
 *
 * @param data Pipcook uniform data
 * @param model Eshcer model
 */
const modelTrain = async (option: Record<string, any>, api: any, model: any): Promise<UniModel> => {
  const {
    modelPath,
    mode = 'cn'
  } = option;
  const { boa } = api;
  const sys = boa.import('sys');

  sys.path.insert(0, path.join(__dirname, 'assets'));

  const classifier = model;

  const rawData = [];
  const rawClass = [];
  let sample = await api.dataSource.nextTrain();
  while (sample) {
    rawData.push(sample.data);
    rawClass.push(sample.label);
    sample = await api.dataSource.nextTrain();
  };
  const text_list = TextProcessing(rawData, rawClass, boa);

  let stopWords = mode === 'en' ? en : cn;
  const stopwords_set = await MakeWordsSet(stopWords);
  const feature_words = words_dict(text_list[0], stopwords_set);
  const feature_list = TextFeatures(text_list[1], feature_words, boa);
  classifier.fit(feature_list, text_list[2]);
  await fs.writeFile(path.join(modelPath, 'stopwords.txt'), stopWords);
  save_all_words_list(feature_words, path.join(modelPath, 'feature_words.pkl'), boa);
  saveBayesModel(classifier, path.join(modelPath, 'model.pkl'), boa);
  return classifier;
};

const main = async(options: Record<string, any>, api: any) => {
  let model = await modelDefine(options, api);
  model = await modelTrain(options, api, model);
};
export default main;
