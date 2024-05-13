import numpy as np

def calculate_class_average(scores):
    subject_avg = {}
    for student, subjects in student_scores.items():
        for subject, scores in subjects.items():
            if subject not in subject_avg:
                subject_avg[subject] = []
            subject_avg[subject].extend(scores)
    avg_scores = {}
    for subject, scores in subject_avg.items():
        avg_score = np.mean(scores)
        avg_scores[subject] = avg_score

    return avg_scores

def compare_to_national_average(class_avg, national_avg):
    comparison = {}
    for course, class_mean in class_avg.items():
        national_mean = national_avg[course]['mean']
        difference = class_mean - national_mean
        comparison[course] = {'class_mean': class_mean, 'national_mean': national_mean, 'difference': difference}
    return comparison

def identify_outliers(scores, threshold):
    outliers = {}
    for student, subjects in student_scores.items():
        for subject, scores in subjects.items():
            # calculatethe z-scores for each score
            z_scores = (scores - np.mean(scores)) / np.std(scores)
            # finding index of outliers based on threshold
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
            if len(outlier_indices) > 0:
                if student not in outliers:
                    outliers[student] = {}
                outliers[student][subject] = [scores[i] for i in outlier_indices]

    return outliers

#sample data
student_scores = {
    'A': {'Math': [80, 85, 90], 'Science': [10, 70]},
    'B': {'Math': [90, 85, 95], 'Science': [80, 85]},
    'C': {'Math': [75, 80, 85], 'Science': [70, 65]}
}

national_averages = {
    'Math': {'mean': 85, 'std_dev': 5},
    'Science': {'mean': 75, 'std_dev': 5}
}

#calculate class avg
class_avg = calculate_class_average(student_scores)

#compare class avg to national avg
comparison = compare_to_national_average(class_avg, national_averages)

#iddentify outliers among students
outliers = identify_outliers(student_scores, threshold=5)

print("Class Average Scores:")
for course, mean in class_avg.items():
    print(f"{course}: {mean}")

print("\nComparison with National Average:")
for course, data in comparison.items():
    print(f"{course}: Class Mean - {data['class_mean']}, National Mean - {data['national_mean']}, Difference - {data['difference']}")

print("\nOutliers:")
for student, tests in outliers.items():
    print(f"{student}: {tests}")