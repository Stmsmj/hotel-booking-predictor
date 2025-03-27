# hotel booking predictor

in this i tried to find the best model for a imbalanced dataset with 36 million rows. for this dataset we are gonna do some classification. unfortunately link of this dataset is not accessible at the time of writing this. 
main challenges of this code was:
* data was imbalanced
* data wasnt representative of what we are asked for

for preprocessing of this code i did lots of visualizations and based on some of them i removed what seemed to be outliers according to its z-score or IQR. after removing outliers i tried to tackle issues caused by being a imbalanced dataset.
i tried different solutions and tried to find a good threshold for our predictions. with our final dataset we going after checking different models to see which will perform the best.

lets look at dataset:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>search_date</th>
      <th>channel</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>destination</th>
      <th>checkIn_date</th>
      <th>checkOut_date</th>
      <th>n_adults</th>
      <th>n_children</th>
      <th>n_rooms</th>
      <th>hotel_category</th>
      <th>is_booking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>u461899</td>
      <td>2019-01-07 00:00:02</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d669</td>
      <td>2019-03-14</td>
      <td>2019-03-15</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>g41</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>u13796</td>
      <td>2019-01-07 00:00:06</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d8821</td>
      <td>2019-01-19</td>
      <td>2019-01-26</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>g58</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>u1128575</td>
      <td>2019-01-07 00:00:06</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d25064</td>
      <td>2019-01-19</td>
      <td>2019-01-22</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>g91</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>u1080476</td>
      <td>2019-01-07 00:00:09</td>
      <td>c9</td>
      <td>False</td>
      <td>True</td>
      <td>d7635</td>
      <td>2019-05-29</td>
      <td>2019-06-05</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>g10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u1080476</td>
      <td>2019-01-07 00:00:17</td>
      <td>c9</td>
      <td>False</td>
      <td>True</td>
      <td>d7635</td>
      <td>2019-05-29</td>
      <td>2019-06-05</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>g10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34742970</th>
      <td>u553256</td>
      <td>2020-11-30 23:59:48</td>
      <td>c2</td>
      <td>True</td>
      <td>True</td>
      <td>d45532</td>
      <td>2020-12-07</td>
      <td>2020-12-08</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>g48</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34742971</th>
      <td>u529472</td>
      <td>2020-11-30 23:59:49</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d8279</td>
      <td>2020-12-27</td>
      <td>2021-01-02</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>g18</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34742972</th>
      <td>u18236</td>
      <td>2020-11-30 23:59:53</td>
      <td>c4</td>
      <td>False</td>
      <td>False</td>
      <td>d20275</td>
      <td>2021-04-22</td>
      <td>2021-04-25</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>g5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34742973</th>
      <td>u10888</td>
      <td>2020-11-30 23:59:54</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d19371</td>
      <td>2020-12-29</td>
      <td>2020-12-30</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>g17</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34742974</th>
      <td>u233344</td>
      <td>2020-11-30 23:59:55</td>
      <td>c9</td>
      <td>False</td>
      <td>False</td>
      <td>d22862</td>
      <td>2021-08-16</td>
      <td>2021-08-18</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>g44</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>34742975 rows × 13 columns</p>
</div>
