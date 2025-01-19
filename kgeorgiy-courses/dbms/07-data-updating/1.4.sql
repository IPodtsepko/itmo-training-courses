delete from
    Students
where
    StudentId in (
        select
            StudentId
        from
            Marks
        group by
            StudentId
        having
            Count(Mark) >= 3
    );
