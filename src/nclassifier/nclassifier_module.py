# @title Библиотеки
import numpy as np
import pandas as pd
import os


# In[3]:


def get_class(data:np.ndarray|pd.DataFrame, classifier:str='SHEPARD', show_graph:bool=False, code_indexes:bool = False, class_dict:dict=None, global_path:str = '')->pd.Series:
      '''extension
      По данным массива определяет классификацию

      :data: 2D массив np.ndarray (n,m) или pd.DataFrame где m - это компоненты последоватльно в зависимости от типа классификатора
      :classifier: Тип классификатора или путь к *.npz файлу пользователя:
                "SHEPARD": классификация осадков по содержанию в них глины(аргиллит),песка  и алеврита  (Shерагd. Journ. Seel. Petrol., 24, p. 157, Fig. 7. 1954).
      :show_graph: Расширение файлов (регистронезависимое). По умолчанию False
      :code_indexes: В качестве индексов возвращает коды. По умолчанию False
      :class_dict: Словарь кодов
      :global_path: путь к файлам классификаторам
      :return: Словарь {абсолютный_путь_директории: [список_файлов]}
      '''

      def _array_to_class_dict(class_arr:np.ndarray)->dict:
          return dict(class_arr.tolist())

      def _get_points3(data:np.ndarray, classifier_shape:tuple, param_type:int, show_graph:bool, code_indexes:bool, class_dict:dict)->pd.Series:

            height, width = classifier_shape

            # Расчет координат для треугольной диаграммы
            clay = data[:, 0]
            sand = data[:, 1]
            silt = data[:, 2]

            # Y: от верхней точки (0% глины) к основанию (100% глины)
            Y = (height - 1) * (1 - clay / 100)

            # X: зависит от соотношения песок/алеврит с учетом сужения к вершине
            valid = (sand + silt) > 0
            ratio = np.zeros_like(silt)
            ratio[valid] = silt[valid] / (sand[valid] + silt[valid])

            # Преобразование в координаты треугольника
            X = (ratio * (width - 1) * (1 - clay / 100)) + ((width - 1) * (clay / 100) / 2)

            # Ограничение координат в пределах массива
            Y =  np.clip(Y, 0, height - 1).astype(int)
            X = np.clip(X, 0, width - 1).astype(int)

            return X, Y

      classifier = os.path.join(global_path,f'{classifier}.npz') if len(classifier.split('.'))==1 else classifier
      classifier_arr = np.load(classifier)
      classifier_data = classifier_arr['classifier']
      class_dict = _array_to_class_dict(classifier_arr['attrs']) if class_dict==None else class_dict
      param_count = classifier_arr['params'][0]
      param_type = classifier_arr['params'][1]
      name = classifier_arr['name'][0]
      label = classifier_arr['label'][0]
      function = locals()[f'_get_points{param_count}'] # Получаем локальноую функцию по имени

      height, width = classifier_data.shape  # (3998, 4616) - не обязательно такие размеры, все зависит от выбранной точности при создании файла дискретного классификатора

      data = pd.DataFrame(data)
      data = data.dropna(how='any')

      indexes = data.index
      data = np.asarray(data, dtype=np.float64)

      if len(data.shape)!=2 or data.shape[1]!=param_count: # Провека на соответсвие, что это 2D массив и что количество параметров равно param_count - количества параметров для классификации
        return None

      # Нормализация до 100% (глина, песок, алеврит)
      data = 100 * data / (data.sum(axis=1, keepdims=True) + 1e-10)

      X, Y  = function(data, (height, width),param_type, show_graph, code_indexes, class_dict)

      if show_graph:
          import matplotlib.pyplot as plt
          from matplotlib.patches import Polygon
          plt.figure(figsize=(5, 5))
          ax = plt.axes()
          ax.imshow(classifier_data)

          ax.scatter(X, Y, c='red', s=20, alpha=0.8)
          for i, (txt, idx) in enumerate(zip(np.arange(1,data.shape[0]+1),indexes)):
            plt.annotate(f'{txt}({idx})', (X[i], Y[i]))
          plt.title(name, pad=20)
          labels = list(class_dict.values())

          if param_count==3:
            # Получим текущие оси и нарисуем боковые грани
            axes = plt.gca()
            axes.set_aspect("equal")
            axes.add_patch(Polygon(([width-1,height-1],[(width-1)/2,0], [0,height-1], [0,0],[width-1,0]), alpha=0.8, fc='w'))

            plt.xlabel(f"{labels[2]} ↔ {labels[1]}")
            plt.ylabel(f"{labels[0]}")

          plt.xticks([])
          plt.yticks([])
          plt.show()

      # Получение классов
      classes = classifier_data[Y, X]
      if code_indexes:
        result = pd.Series(class_dict)[classes]
      else:
        result = pd.Series(class_dict)[classes].reset_index(drop=True)
        result.index = indexes
      result.name = label
      return result


