-- Наивысшая оценка по курсу, заданному идентификатором.
select
    max(Mark) as MaxMark
from
    Marks
where
    CourseId = :CourseId;
