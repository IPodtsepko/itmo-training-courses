update
    Students
set
    Marks = (
        select
            count(Mark)
        from
            Marks
        where
            StudentId = :StudentId
    )
where
    StudentId = :StudentId;
