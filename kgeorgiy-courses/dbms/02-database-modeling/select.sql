select Students.first_name,
       Students.last_name,
       Subjects.name as subject,
       Teachers.last_name as teacher,
       mark
from Marks
natural join Students
natural join Subjects
join Teachers on (Marks.teacher_id = Teachers.teacher_id)
order by student_id;
