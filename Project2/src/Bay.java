import java.util.LinkedList;
import java.util.Queue;

public class Bay {
    private Queue<Car> waiting_list;
    private int car_wash_time;
    private int washer_remaining_time;
    private boolean available;


    public Bay(int washing_minute){
        waiting_list = new LinkedList<>();
        car_wash_time = washing_minute;
        washer_remaining_time = car_wash_time;
        available = true;
    }

    public Queue<Car> get_waiting_list() {
        return waiting_list;
    }
    public void add_waiting_list(Car car) {
        waiting_list.add(car);
    }
    public Car get_from_waiting_list(){
        return waiting_list.poll();
    }
    public boolean isAvailable() {
        return available;
    }
    public void make_washer_unavailable() {
        available = false;
    }
    public void make_washer_available() {
        available = true;
        washer_remaining_time = car_wash_time;
    }
    public void decrement_washer_remaining_time(){
        if(washer_remaining_time > 0){
            washer_remaining_time -= 1;
        } else {
            this.make_washer_available();
        }
    }

}
