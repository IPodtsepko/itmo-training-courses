-- Поиск курса по префиксу названия.
select
    CourseId,
    CourseName
from
    Courses
where
    CourseName like 'Базы%';
