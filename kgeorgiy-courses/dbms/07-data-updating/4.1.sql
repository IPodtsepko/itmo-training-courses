insert into
    Marks (StudentId, CourseId, Mark)
select
    StudentId,
    CourseId,
    Mark
from
    NewMarks
where
    not exists (
        select
            *
        from
            Marks
        where
            Marks.StudentId = NewMarks.StudentId
            and Marks.CourseId = NewMarks.CourseId
    );
