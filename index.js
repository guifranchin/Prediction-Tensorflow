const tensorflow = require("@tensorflow/tfjs");
const csv = require("csv-parser");
const fs = require("fs");


(() => {
  const trainingData = [];
  fs.createReadStream("training_data.csv")
    .pipe(csv())
    .on("data", (row) => {
      trainingData.push(row);
    })
    .on("end", () => {
     
      const inputs = trainingData.map(
        (row) => row.input1 === "TRUE" || row.input1 === "true"
      );
      const labels = trainingData.map(
        (row) => row.label === "TRUE" || row.label === "true"
      );

      
      const [inputsTrain, inputsValidation] = tensorflow.split(
        tensorflow.tensor2d(inputs, [inputs.length, 1]),
        0.8
      );
      const [labelsTrain, labelsValidation] = tensorflow.split(
        tensorflow.tensor2d(labels, [labels.length, 1]),
        0.8
      );

  
      const numUnitsList = [8, 16, 32, 64];
      const numLayersList = [1, 2, 3, 4];
      for (const numUnits of numUnitsList) {
        for (const numLayers of numLayersList) {
    
          const model = tensorflow.sequential();

         
          for (let i = 0; i < numLayers; i++) {
            model.add(
              tensorflow.layers.dense({ units: numUnits, inputShape: [1] })
            );
          }
          model.add(
            tensorflow.layers.dense({ units: 1, activation: "sigmoid" })
          );
          model.compile({ optimizer: "adam", loss: "binaryCrossentropy" });

        
          model.fit(inputsTrain, labelsTrain, {
            epochs: 10,
            validationSplit: 0.2,
            callbacks: {
              onEpochEnd: (epoch, log) =>
                console.log(`Época ${epoch}: loss = ${log.loss}`),
            },
          });

         
          const valAcc = model.evaluate(inputsValidation, labelsValidation);

          console.log(
            `Configuração de hiperparâmetros: ${numLayers} camadas ${numUnits} unidades por camada`
          );
          console.log(`Acurácia de validação: ${valAcc}`);
        }
      }
    });
})();
