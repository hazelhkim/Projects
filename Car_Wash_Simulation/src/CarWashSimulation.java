import java.util.Scanner;
import java.util.Random;
public class CarWashSimulation {

    private static float[] run_simulation(int minute){
        Bay bay = new Bay(minute);
        float total_waiting_time = 0, total_car_arrived = 0;
        int long_wait_count = 0, count_serviced = 0;
        // for each minute of the simulation till 600 mins.
        for(int i = 0; i < 600; i++) {
            //if a car arrives
            Random random = new Random();
            int chance = random.nextInt(4);
            if(chance == 0) {
                Car car = new Car(i);
                bay.add_waiting_list(car);
                total_car_arrived ++;
                //System.out.println("time step " + i + " : A car arrived." );
                //System.out.println("Car arrived at time step" + i + "." );
            }
                //System.out.println("time step " + i + " : No car arrived.");

            if(bay.isAvailable()&& !bay.get_waiting_list().isEmpty()) {
                Car car_washing = bay.get_from_waiting_list();
                count_serviced ++;
                int each_waiting_time = i - car_washing.get_arrival_time();
                total_waiting_time += each_waiting_time;
                //System.out.println("Car No."+count_serviced+ "has waited for" + (i - car_washing.get_arrival_time()) + "mins.");

                if( each_waiting_time > 10 ) {
                    long_wait_count ++;
                    //System.out.println(each_waiting_time);
                }
                bay.make_washer_unavailable();
            } else if (!bay.isAvailable()) {
                bay.decrement_washer_remaining_time();
            }
        }
        while(!bay.get_waiting_list().isEmpty()){
            int waiting_time = 599 - bay.get_from_waiting_list().get_arrival_time();
            total_waiting_time += waiting_time;
            if(waiting_time > 10){
                long_wait_count ++;
            }
        }
        float avg = total_waiting_time/total_car_arrived;
        // minute/ avg/ total_cars/ cars_serviced/ cars_long_waited.
        float[] ans = new float[5];
        ans[0] = minute;
        ans[1] = avg;
        ans[2] = total_car_arrived;
        ans[3] = count_serviced;
        ans[4] = long_wait_count;

        return ans;
    }

    public static void main(String[] args) {

        //Scanner scan = new Scanner(System.in); //Reading from System.in
        //System.out.println("Enter the number of minutes that the Car Wash Machine would take: ");
        //int minute = scan.nextInt(); //Scans the next token of the input as an int.
        //scan.close();

        float[][] result = new float[2][5];
        for(int i = 0; i < 7; i++){
            float[] ans1 = run_simulation(4);
            float[] ans2 = run_simulation(3);
            for(int j = 0; j<ans1.length;j++){
                result[0][j] += ans1[j];
                result[1][j] += ans2[j];
            }
        }
        for(int j = 0; j<result[0].length;j++){
            result[0][j] = result[0][j]/7;
            result[1][j] = result[1][j]/7;
        }
        // minute/ avg/ total_cars/ cars_serviced/ cars_long_waited.
        System.out.println("minute | average_waiting_time | total_cars_arrived | cars_serviced | cars_long_waited");
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                System.out.print(result[i][j] + "\t  | \t");
            }
            System.out.println();
        }
    }
}