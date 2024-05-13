from datasets.dataset_noxGroundTruth import data

class NoxGroundTruth:
    def __init__(self, data):
        self.data = data

    def num_stations(self):
        return len(self.data)

    def nox_reading_5th_station(self):
        return self.data[4][1]

    def nox_reading_last_station(self):
        return self.data[-1][1]

    def sum_nox_w_stations(self):
        return sum(station[1] for station in self.data if station[0][0].lower() == 'w')

    def average_nox_reading(self):
        num_stations = self.num_stations()
        total_nox_with_error = sum(station[1] * (1 + station[2] / 100) for station in self.data)
        return total_nox_with_error / num_stations if num_stations != 0 else 0

nox_analyzer = NoxGroundTruth(data)

# printing results
print("Total number of stations:", nox_analyzer.num_stations())
print("NOx reading from the 5th station:", nox_analyzer.nox_reading_5th_station())
print("NOx reading from the last station:", nox_analyzer.nox_reading_last_station())
print("Total sum of NOx on stations with names beginning with 'W':", nox_analyzer.sum_nox_w_stations())
print("Average NOx reading considering maximum error:", nox_analyzer.average_nox_reading())
