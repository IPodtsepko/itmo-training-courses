update
    Marks
set
    Mark = (
        select
            Mark
        from
            NewMarks
        where
            NewMarks.StudentId = Marks.StudentId
            and NewMarks.CourseId = Marks.CourseId
    )
where
    exists (
        select
            *
        from
            NewMarks
        where
            NewMarks.StudentId = Marks.StudentId
            and NewMarks.CourseId = Marks.CourseId
    );
