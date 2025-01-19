update
    Students
set
    Marks = Marks + (
        select
            count(Mark)
        from
            NewMarks
        where
            NewMarks.StudentId = Students.StudentId
    );
