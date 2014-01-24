package lda;
import java.util.ArrayList;
import Jama.Matrix;

public class LDA {
	private double[][] groupMean;
	private double[][] pooledInverseCovariance;
	private double[] probability;
	private ArrayList<Integer> groupList = new ArrayList<Integer>();
        static int hasil;
        static double f1,f2,f3;
	
        public LDA(){

        }
	public LDA(double[][] d, int[] g, boolean p) {
		// memeriksa apakah data dan kelompok array mempunyai ukuran yang sama
		if (d.length != g.length)
			return;

		double[][] data = new double[d.length][d[0].length];//panjang data(i) dan fitur(j)
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[i].length; j++) {
				data[i][j] = d[i][j];
			}
		}
		int[] group = new int[g.length];
		for (int j = 0; j < g.length; j++) {
			group[j] = g[j];
		}

		double[] globalMean;
		double[][][] covariance;

		//memisahkan berdasarkan grup atau kelas
		for (int i = 0; i < group.length; i++) { 
			if (!groupList.contains(group[i])) {
				groupList.add(group[i]);
			}
		}

		//membagi data ke dalam subset
		ArrayList<double[]>[] subset = new ArrayList[groupList.size()];
		for (int i = 0; i < subset.length; i++) {
			subset[i] = new ArrayList<double[]>();
			for (int j = 0; j < data.length; j++) {
				if (group[j] == groupList.get(i)) {
					subset[i].add(data[j]);
				}
			}
		}

		//menghitung mean tiap fitur tiap kelas
		groupMean = new double[subset.length][data[0].length];
		for (int i = 0; i < groupMean.length; i++) {
			for (int j = 0; j < groupMean[i].length; j++) {
				groupMean[i][j] = getGroupMean(j, subset[i]);
			}
		}

		//menghitung global mean atau mean tiap fitur pada semua kelas
		globalMean = new double[data[0].length];
		for (int i = 0; i < data[0].length; i++) {
			globalMean[i] = getGlobalMean(i, data);
		}

		// correct subset data
		for (int i = 0; i < subset.length; i++) {
			for (int j = 0; j < subset[i].size(); j++) {
				double[] v = subset[i].get(j);

				for (int k = 0; k < v.length; k++)
					v[k] = v[k] - globalMean[k];//zero mean = data ke i- global mean

				subset[i].set(j, v);
			}
		}

		//menghitung kovarian,kovarian=(matriks zero mean transpose*matriks zero mean)/jml data
		covariance = new double[subset.length][globalMean.length][globalMean.length];
		for (int i = 0; i < covariance.length; i++) {
			for (int j = 0; j < covariance[i].length; j++) {
				for (int k = 0; k < covariance[i][j].length; k++) {
					for (int l = 0; l < subset[i].size(); l++)
						covariance[i][j][k] += (subset[i].get(l)[j] * subset[i]
								.get(l)[k]);

					covariance[i][j][k] = covariance[i][j][k]
							/ subset[i].size();
				}
			}
		}

		//menghitung kovarian global
		pooledInverseCovariance = new double[globalMean.length][globalMean.length];
		for (int j = 0; j < pooledInverseCovariance.length; j++) {
			for (int k = 0; k < pooledInverseCovariance[j].length; k++) {
				for (int l = 0; l < subset.length; l++) {
					pooledInverseCovariance[j][k] += ((double) subset[l].size() / (double) data.length)//jumlah kelas dibagi jumlah data dikali kovarian
							* covariance[l][j][k];
				}
			}
		}
                //mencari invers matriks kovarian
		pooledInverseCovariance = new Matrix(pooledInverseCovariance).inverse()
				.getArray();

		//menghitung probability posterior untuk kelas yang berbeda
		this.probability = new double[subset.length]; 
		if (!p) {
			double prob = 1.0d / groupList.size();
			for (int i = 0; i < groupList.size(); i++) {
				this.probability[i] = prob;
			}
		} else {
			for (int i = 0; i < subset.length; i++) {
				this.probability[i] = (double) subset[i].size()//lemon=3/15, manis=5/15, nipis=7/15
						/ (double) data.length;
			}
		}
	}

	private double getGroupMean(int column, ArrayList<double[]> data) {
		double[] d = new double[data.size()];
		for (int i = 0; i < data.size(); i++) {
			d[i] = data.get(i)[column];
		}

		return getMean(d);
	}

	private double getGlobalMean(int column, double data[][]) {
		double[] d = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			d[i] = data[i][column];
		}

		return getMean(d);
	}

        //menghitung nilai fungsi discriminant untuk kelas yang berbeda
	public double[] getDiscriminantFunctionValues(double[] values) {
		double[] function = new double[groupList.size()];
		for (int i = 0; i < groupList.size(); i++) {
			double[] tmp = matrixMultiplication(groupMean[i],
					pooledInverseCovariance);
			function[i] = (matrixMultiplication(tmp, values))//fi=miu i*invers kovarian*data testing-1/2 miu i*invers kovarian*miu i trans+ln(pi)
					- (.5d * matrixMultiplication(tmp, groupMean[i]))
					+ Math.log(probability[i]);
		}

		return function;
	}

        //memprediksi masuk kelas mana
	public int predict(double[] values) {
		int group = -1;
		double max = Double.NEGATIVE_INFINITY;
		double[] discr = this.getDiscriminantFunctionValues(values);
		for (int i = 0; i < discr.length; i++) {
			if (discr[i] > max) {
				max = discr[i];
				group = groupList.get(i);
			}
		}

		return group;
	}

	//mengalikan dua matriks
	private double[] matrixMultiplication(double[] matrixA, double[][] matrixB) {

		double c[] = new double[matrixA.length];
		for (int i = 0; i < matrixA.length; i++) {
			c[i] = 0;
			for (int j = 0; j < matrixB[i].length; j++) {
				c[i] += matrixA[i] * matrixB[i][j];
			}
		}

		return c;
	}
	
        private double matrixMultiplication(double[] matrixA, double[] matrixB) {

		double c = 0d;
		for (int i = 0; i < matrixA.length; i++) {
			c += matrixA[i] * matrixB[i];
		}

		return c;
	}

	public static double getMean(final double[] values) {
		if (values == null || values.length == 0)
			return Double.NaN;

		double mean = 0.0d;

		for (int index = 0; index < values.length; index++)
			mean += values[index];

		return mean / (double) values.length;
	}

	public static void test(extraksi_fitur e,double a,double b,double c,double d) {
            extraksi_fitur ef=e;
		int[] group = { 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 };//1=lemon,2=manis,3=nipis
		double[][] data = new double[15][4];//15=jumlah data,4=fitur R,G,B,D
                int count=0;
                for(int i=0;i<3;i++){//kelas lemon ada  3
                    data[count][0]=ef.fiturlemon[i][0];//[0]=R  [1]=G  [2]=B   [3]=D
                    data[count][1]=ef.fiturlemon[i][1];
                    data[count][2]=ef.fiturlemon[i][2];
                    data[count++][3]=ef.fiturlemon[i][3];
                }
                for(int i=0;i<5;i++){//kelas manis ada 5
                    data[count][0]=ef.fiturmanis[i][0];
                    data[count][1]=ef.fiturmanis[i][1];
                    data[count][2]=ef.fiturmanis[i][2];
                    data[count++][3]=ef.fiturmanis[i][3];
                }
                for(int i=0;i<7;i++){//kelas nipis ada 7
                    data[count][0]=ef.fiturnipis[i][0];
                    data[count][1]=ef.fiturnipis[i][1];
                    data[count][2]=ef.fiturnipis[i][2];
                    data[count++][3]=ef.fiturnipis[i][3];
                }
		LDA test = new LDA(data, group, true);
		double[] testData = {a,b,c,d};

		//test
		double[] values = test.getDiscriminantFunctionValues(testData);
		for(int i = 0; i < values.length; i++){
			System.out.println("Discriminant function " + (i+1) + ": " + values[i]);
		}

		System.out.println("Predicted group: " + test.predict(testData));
                hasil=test.predict(testData);
                f1=values[0];
                f2=values[1];
                f3=values[2];
	}
}
